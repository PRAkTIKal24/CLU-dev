"""
Central configuration management for CHLU.

This module defines all configurable parameters using dataclasses,
providing type safety and defaults for all experiments, training, and models.
"""

from dataclasses import asdict, dataclass, field
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
    # Learning-rate multiplier for the inertial-mass parameters (log_mass leaves
    # of every unit): the optimizer runs those leaves at learning_rate *
    # mass_lr_mult on their own Adam slot (optax.multi_transform), the rest at
    # learning_rate. Motivation (critique P5/G4): three corroborations report
    # that learned M never differentiates into a hierarchy ("designed-in or
    # induced" doctrine), but log_mass was never given its own lr — while the
    # gamma_phi(q) two-timescale work proved this exact class of q-space-adjacent
    # parameter cannot move at the base Adam lr. Default 1.0 = bit-compatible
    # (log_mass shares the base slot, identical to plain optax.adam); a lattice
    # of units shares one multiplier. Inert for kinetic_mode="newtonian_identity"
    # (H never reads log_mass => zero gradient => zero Adam update at any lr).
    mass_lr_mult: float = 1.0
    # V(data)-energy anchor (sleep-erosion cure; anchor-robustness P11). Adds a
    # wake term lambda * (mean_i V(anchor_i) - target)^2 to the loss, where the
    # anchor points are the data q's at the window start (data[:, 0, :dim]) and
    # target = the epoch-0 mean V(anchor) captured from the initial model. This
    # PINS the potential's value on the data manifold, preventing the wake-sleep
    # CD sleep phase from inverting a DESIGNED degenerate vacuum along its flat
    # (Goldstone) direction — the failure the wake MSE cannot see (handover
    # §7.14 / anchor-robustness). Envelope (exp-d SO(2), CD, f5/s500, 3000 ep,
    # 5 seeds): lambda=0 destroys the vacuum 5/5; lambda in {1,10,100} hold it;
    # lambda=100 is seed-bulletproof (r*=0.911+-0.016) at ~35x wake-MSE cost,
    # lambda~=10 gives the strongest noise rejection but 1/5 seeds can collapse.
    # Default 0.0 = OFF = bit-compatible with all prior runs (term not added).
    # Orthogonal to volume conservation: does NOT rescue a non-symplectic vacuum.
    anchor_data_energy_lambda: float = 0.0

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
    # F5 Prop-9) or "fdt" = per-mode sigma_i* = sqrt(M_eff_i*T*gamma*(2-gamma)).
    #
    # "fdt" is the exact discrete fluctuation-dissipation noise — temperatures
    # in energy units, stationary law exp(-H/T) — **only for the Newtonian
    # kinetic modes** (newtonian_identity / newtonian_learned). In
    # kinetic_mode="relativistic" NO sigma gives a Gibbs invariant (CM-17):
    # the coded O-step p<-(1-gamma)p+sigma*xi is a linear OU recursion whose
    # stationary momentum law is Gaussian, while relativistic Gibbs demands
    # Maxwell-Juttner. Root cause: the Gibbs-preserving underdamped Langevin
    # damps the *velocity* grad_p T, this code damps *p* — the same thing iff
    # T is Newtonian (Gamma = gamma*M). The defect is governed by d*Theta
    # (Theta = T/(m0*c^2)), NOT Theta alone (see CHLU.gibbs_defect_parameter);
    # Exp-C runs at d=784, Theta=1 => d*Theta=784. Raising c/rest_mass is NOT a
    # free fix at that d; the exact fix is langevin_noise="fdt_relativistic"
    # (latent-mass thermostat). CHLU.stochastic_step warns (does not raise) on
    # relativistic+fdt.
    #
    # Default "legacy" preserves behavior for existing checkpoints/schedules.
    # NOTE: under "legacy" T is NOT in energy units (dt and M_eff are absorbed)
    # and there is no Gibbs invariant in any kinetic mode — any temperature
    # claim must state this flag.
    langevin_noise: str = "legacy"

    # --- Friction field gamma_phi(q) (trash regions; F5 Def-5/Prop-11) ---
    # "none" (default; scalar-gamma damping, bit-compatible with all prior
    # behavior), "fixed" (hand-placed frozen holes — oracle/control arms),
    # "learned" (contrastively trained holes: wake protects data, sleep damns
    # hallucinations — brainstorm Thread 1).
    friction_field: str = "none"
    friction_field_k: int = 1  # number of holes K
    friction_field_gamma_max: float = 0.5  # strict cap: gamma_phi in [0, gamma_max)
    friction_field_width: float = 0.25  # horizon width w
    # Horizon gate shape: "sigmoid" (default; smooth, infinite tail) or
    # "compact" (smoothstep with an EXACT hard cutoff at each hole radius r_k —
    # gamma_phi identically 0 beyond r_k, closing the sigmoid tail-leakage
    # retention gap seen for learned arms in the S1 pilot; gamma-field-build
    # follow-up 2).
    friction_field_gate: str = "sigmoid"
    friction_field_init_radius: float = 1.0  # hole radius at init ("learned")
    friction_field_init_strength: float = 0.15  # hole strength gamma_k at init
    friction_field_init_center_scale: float = 1.0  # centers ~ N(0, scale^2)
    # Hand-placed centers for "fixed" mode: list of [dim]-lists; None -> origin
    friction_field_fixed_centers: Optional[List[List[float]]] = None
    friction_field_fixed_radius: float = 0.6  # hole radius for "fixed"
    friction_field_fixed_strength: float = 0.3  # hole strength for "fixed"
    # Contrastive-training weights (only active when the model carries a field)
    friction_field_protect_lambda: float = 1.0  # wake: push gamma_phi(q_data) down
    friction_field_hallu_lambda: float = 1.0  # sleep: push gamma_phi(q_hallu) up
    # Which negatives count as hallucinations (Thread-1 says "persistent"):
    # "energy" (default) weights each evolved negative by how far its energy
    # sits above the current wake window's band, sigmoid((H - max_H_data)/std)
    # — CD negatives that converged into the data band get ~no friction vote
    # (ungated, they drag friction onto the data manifold, fighting the
    # protection term; observed in the S1 smoke run). "all" = ungated.
    friction_field_hallu_gate: str = "energy"
    # Optional C1 ablation (mo-deep-read §5): nudge gamma_k -> 2*dt*mu(c_k),
    # the critical-damping forgetting optimum. 0.0 = OFF (measure, don't force).
    friction_field_c1_lambda: float = 0.0
    # Learning rate for the field's own parameters (two-timescale training):
    # hole centers live in q-space and must travel O(units), but Adam caps
    # parameter velocity at ~lr/step — at the base lr the placement cannot
    # move at pilot scale (observed in the S1 smoke run). None = base lr.
    friction_field_lr: Optional[float] = 1e-2
    # --- Adaptive-K (gamma-field-build follow-up 1) ---
    # When True, the sleep phase spawns a new hole where persistent
    # (energy-gated) hallucination density accumulates beyond a threshold, and
    # prunes holes whose strength decays below a floor. Fixes the S1 finding
    # that a lone sigmoid hole far from all negatives gets no placement
    # gradient (gradient locality => locus discovery 2/6). Default False keeps
    # the fixed-K behavior. Structural edits reset the optimizer state.
    friction_field_adaptive_k: bool = False
    friction_field_max_k: int = 8  # hard cap on spawned holes
    # Decayed uncovered energy-gated weight at which to spawn (per sleep event)
    friction_field_spawn_threshold: float = 5.0
    friction_field_spawn_min_dist: float = 0.5  # min distance from existing holes
    friction_field_spawn_radius: float = 0.5  # radius of a spawned hole
    friction_field_spawn_strength: float = 0.15  # strength gamma_k of a spawned hole
    friction_field_prune_floor: float = 0.02  # prune holes with gamma_k below this


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
    # Default lowered 1000 -> 150 after v2-full-runs Finding 0 (handover §7.14):
    # at 1000 epochs the wake–sleep sleep phase EROSES the designed SO(2)
    # vacuum (8/8 seeds: r*->0, the data ring inverts into a local MAXIMUM,
    # ring depth +0.060@150 -> -0.047@1000 with inversion between 300–600 ep).
    # 150 ep is the engineer-validated regime where the designed vacuum is
    # intact. Set sleep_mode="off" for the wake-only, data-pinned regime
    # (r*=1.0000 through 1000 ep) if you need longer training.
    train_epochs: int = 150
    # Sleep-phase switch for the erosion study. "on" (default) = standard
    # wake–sleep contrastive training. "off" = wake-only (sleep_frequency->inf):
    # the vacuum is data-pinned and does not erode (v2-full-runs Finding 0).
    sleep_mode: str = "on"
    use_pretrained: bool = False
    kinetic_energy_mode: str = (
        "newtonian_learned"  # learned M => isotropy falsifiable is live
    )
    potential_type: str = (
        "so2_invariant"  # "so2_invariant" (designed) or "mlp" (emergent)
    )
    tie_channel_mass: bool = (
        True  # kinetic isotropy (F5 §4.1); False = broken-isotropy switch
    )
    tilt_delta: float = (
        0.0  # explicit breaking delta*cos(n*theta) (GMOR probe, F5 §3.3c)
    )
    tilt_n: int = 1
    # Linear AMBIENT spurion -delta*(u.q) along the channel direction u (the
    # ChPT quark-mass term). Unlike the angular tilt above it lets the vacuum
    # radius r* run with delta, resolving mu^2, F^2 = M_ch*r*^2 and the
    # condensate Sigma = r* independently => GMOR proper (mu^2 F^2 = delta*Sigma
    # exactly). 0.0 (default) = no spurion, behavior unchanged.
    spurion_delta: float = 0.0
    spurion_angle: float = 0.0  # angle (rad) of u in the channel plane
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


@dataclass
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

    # --- v1-pivot: learned calibration head + escalation/abstention ---
    # (exp_v1_calibration; Head decision 2026-07-07: squeeze retries PARKED,
    #  escalation = staged governed relaxation on the gate run's cost ladder.)
    calib_difficulty_levels: List[List[int]] = field(
        default_factory=lambda: [[128, 16], [128, 24], [128, 32]]
    )
    calib_n_seeds: int = 5  # replicate seeds = project.seed + i
    calib_min_trials_per_level: int = 128
    calib_max_episodes_per_level: int = 8
    # Write-time self-test probes: per stored key, jittered cues at these noise
    # scales (cycled); plus impostor cues from unbound keys (label: wrong by
    # definition — the memory holds no binding for them).
    calib_probes_per_key: int = 8
    calib_cue_noise_scales: List[float] = field(
        default_factory=lambda: [0.05, 0.15, 0.3]
    )
    calib_n_impostors: int = 16
    calib_features: str = "r_margin"  # deployed gate: "r" | "margin" | "r_margin"
    calib_l2: float = 1.0  # ridge strength of the head fit
    calib_fit_all_stages: bool = True  # fit on probe states at every ladder stage
    # Escalation ladder: base relax_steps, then n_stages x stage_steps more
    # (defaults reproduce the gate run's cost ladder 300/1200/2100/3000).
    calib_n_stages: int = 3
    calib_stage_steps: int = 900
    calib_p_exit: float = 0.5  # learned-gate exit/abstain threshold on p_wrong
    calib_n_policy_taus: int = 25  # swept-threshold points for compute curves
    calib_risk_targets: List[float] = field(default_factory=lambda: [0.05, 0.10])
    calib_ltt_delta: float = 0.1  # LTT confidence parameter

    # --- v1-hopfield-stress: CLU-vs-Hopfield regime map (Head decision 1b) ---
    # The v1-pivot run showed Hopfield is near-perfect (acc 0.983-1.0) on the
    # vanilla MQAR at kv<=32, so the abstention head-to-head is unwinnable
    # there. This block charts *where the trade actually lives* by stressing
    # both systems and classifying each grid cell (Hopfield-dominant /
    # comparable / CLU-gate-advantage). Two stress mechanics, both fair (CLU
    # and Hopfield see the identical stressed cues/embeddings):
    #   - eval_noise: Gaussian sigma added to the deployment query cue (memory
    #     written from clean patterns; degrades retrieval, not storage).
    #   - correlation: key/value embeddings pulled toward shared cluster
    #     centroids (reduced separation = the classic Hopfield failure mode;
    #     changes stored content -> the memory is retrained per correlation).
    # Defaults are laptop-scale (pilot first; report runtimes).
    # (N, kv) with N >= 3*kv (kv-block 2*kv + kv queries fit the sequence) AND
    # kv < vocab_size/2 (distinct keys/values per half-vocab). To push kv past
    # ~vocab_size/2 - 1 (127 at the default vocab), raise vocab_size too.
    regime_capacity_levels: List[List[int]] = field(
        default_factory=lambda: [[128, 32], [256, 64], [384, 96]]
    )
    regime_stress_axis: str = "correlation"  # "correlation" | "eval_noise"
    regime_stress_grid: List[float] = field(
        default_factory=lambda: [0.0, 0.3, 0.6, 0.9]
    )
    regime_n_seeds: int = 3
    regime_episodes_per_cell: int = 1  # episodes per (capacity, stress, seed)
    regime_n_clusters: int = 8  # centroids for the correlated-embedding stress
    # applied deployment cue noise when the stress axis is NOT eval_noise
    regime_base_eval_noise: float = 0.0
    # applied key correlation when the stress axis is NOT correlation
    regime_base_correlation: float = 0.0
    # classification band: |delta| below this = "comparable" (acc & AURC units)
    regime_comparable_margin: float = 0.03

    # --- v1.1: gate stack on a Hopfield memory (exp_v1_hopfield_gate) ---
    # The Hopfield memory is near-perfect on vanilla MQAR (v1-pivot), so the
    # gate has nothing to rank; these apply a stress so it errs and the
    # calibration-transfer / allocation / LTT metrics carry signal. Default 0
    # = vanilla (degenerate, near-perfect). Correlation reuses the regime-map
    # clustered-embedding stress; eval_noise adds Gaussian sigma to BOTH the
    # self-test probe cues and the deployment cues (matched difficulty).
    hopfield_gate_correlation: float = 0.0
    hopfield_gate_eval_noise: float = 0.0


@dataclass
class ExperimentV1WormholeConfig:
    """Configuration for the V1 wormhole-routing experiment (V1 third pillar).

    A chain lattice of N CHLU associative-memory units (each an EBM over its
    own KV subset). A query arrives at the query unit (unit 0); its answer is
    either local (bound in unit 0) or distant (bound only in the archive unit,
    N-1). An energy-gated sparse non-local edge (built on the lattice's smooth
    coupling machinery) opens as a function of the LOCAL residual energy
    R0 = H_0(settled) - floor_0: high R0 (local retrieval failed => the answer
    is elsewhere) opens the wormhole. The open edge is a gated KEY-channel
    spring that TRANSPORTS the query key to the archive's free key half; the
    archive then relaxes to the matching stored pattern and the answer is read
    at the *terminal* (archive) unit. Low R0 (local hit) keeps the gate closed
    and reads unit 0. Mis-routing a local query (gate wrongly open) pulls the
    archive to a cue it does not store => wrong, so gate selectivity has real
    teeth (F5 §7.4 smooth gate; Def-7 escalation beyond one shell; squeezes
    PARKED). Five arms: local-only / gated-wormhole / dense-always-open /
    chain-multi-hop / calibrated-tau-gate.
    """

    # --- lattice / task ---
    n_units_values: List[int] = field(default_factory=lambda: [4, 8])
    embed_dim: int = 12  # per-token embedding; unit dim = 2 * embed_dim (key||value)
    embed_scale: float = 2.0
    vocab_size: int = 128
    kv_per_unit: int = 3  # stored KV pairs per unit (units hold disjoint pairs)
    n_seeds: int = 2  # replicate seeds = project.seed + i
    trials_per_type: int = 48  # jittered queries per {local, distant} per (N, seed)
    query_cue_noise: float = 0.05  # cue jitter (noisy queries => genuine statistics)

    # --- memory model & PCD "write" (per unit) ---
    kinetic_energy_mode: str = "relativistic"
    potential_type: str = "mlp"  # coercive (confinement) — see F5 Prop-10
    hidden_dim: int = 128
    train_epochs: int = 400
    train_lr: float = 1e-3
    train_batch_size: int = 16
    train_k_steps: int = 50
    train_buffer_capacity: int = 128
    train_friction: float = 0.3
    train_temperature: float = 0.3
    train_input_noise_sigma: float = 0.05

    # --- retrieval / routing ---
    dt: float = 0.05
    relax_steps: int = 250  # local relaxation (phase 1; the routing signal R0)
    route_steps: int = 250  # joint routed relaxation (phase 2)
    governor_sensitivity: float = 0.95
    kappa_wormhole: float = 2.0  # gated key-channel transport strength (direct edge)
    kappa_chain: float = 2.0  # adjacent key-channel coupling (multi-hop substrate)
    # Smooth energy gate g = sigmoid((z - gate_z_threshold)/gate_z_width),
    # z = (R0 - median R0)/(IQR R0 + eps): a LABEL-FREE within-deployment
    # normalization of the residual (the v1-l0-gate finding: raw R is not
    # cross-model comparable). g -> 1 as R0 rises (local failure => route).
    gate_z_threshold: float = 0.0
    gate_z_width: float = 0.7
    gate_route_threshold: float = 0.5  # route (and read the terminal) iff g > this

    # --- calibrated tau-gate arm (Def-7 story minus squeezes) ---
    calib_probes_per_key: int = 8
    calib_cue_noise_scales: List[float] = field(
        default_factory=lambda: [0.05, 0.15, 0.3]
    )
    calib_features: str = "r_margin"  # head input: "r" | "margin" | "r_margin"
    calib_l2: float = 1.0
    calib_p_route: float = 0.5  # route iff head.p_wrong(R0[,margin0]) > this
    # Which units supply the impostor (route=True) probes when FITTING the
    # deployed calibrated head (the impostor-composition study, item 4 of
    # v1-router-baseline). At deployment the only DISTANT queries come from the
    # archive (unit N-1), but the legacy head trains impostors from the whole
    # non-local pool (units 1..N-1); as N grows that pool is dominated by
    # non-archive units, shifting the route boundary and OVER-ROUTING local
    # queries (measured local false-positive 53%). Options:
    #   "all_others"    - units 1..N-1 (default; legacy, bit-compatible)
    #   "archive_only"  - just the archive: matches the deployment distant source
    #                     (the measured fix: local FP 53% -> 7%)
    #   "neighbors_only"- units 1..N-2 (everything EXCEPT the archive; worst-case
    #                     mismatch control)
    impostor_policy: str = "all_others"

    # --- learned-router-MLP arm (P9/V1.2: the boring physics-free baseline) ---
    # A 2-layer MLP on the raw query CUE embedding (no energy, no relaxation) that
    # decides route-or-not. Same sparsity budget as the gated wormhole (a binary
    # route decision + the same direct edge on route); trained write-time on the
    # same own-key/impostor probe set as the calibrated head, but consuming the
    # embedding rather than the energy residual. The FLOPs baseline the energy
    # gate must beat: it can SKIP phase-1 relaxation on routed queries (its
    # decision needs no settle), unlike the residual-driven gates.
    router_hidden_dim: int = 32  # MLP hidden width (parameter-matched, tiny)
    router_epochs: int = 300  # Adam steps for the router fit
    router_lr: float = 3e-3
    router_l2: float = 1e-3  # weight decay on the router MLP
    router_p_route: float = 0.5  # route iff sigmoid(mlp(cue)) > this

    # --- workload realism (P9/V1.2): report cost/accuracy per local:nonlocal mix
    # by reweighting the balanced per-query outcomes (each query is processed
    # independently against a fixed calibration pool, so reweighting is exact).
    workload_mixes: List[List[float]] = field(
        default_factory=lambda: [[0.5, 0.5], [0.8, 0.2], [0.95, 0.05]]
    )

    # --- honest cost accounting (P9/V1.2): a FLOPs model replaces unit-steps ---
    # FLOPs per potential value-and-grad eval ~ flops_grad_factor * MACs of the
    # PotentialMLP (fwd MACs = dim*hidden + hidden*hidden + hidden); a Verlet step
    # evaluates grad_H flops_verlet_grads times per active unit.
    flops_grad_factor: float = 6.0  # fwd(2) + bwd(4) FLOP/MAC for a value-and-grad
    flops_verlet_grads: float = 2.0  # grad_q H evals per velocity-Verlet step

    # --- reference baseline ---
    hopfield_beta: float = 20.0  # modern-Hopfield over the UNION of all patterns


@dataclass
class ExperimentLatticeConfig:
    """Configuration for the CLU-lattice experiment (V3 first build, F5 §7).

    A joint-Hamiltonian lattice of CHLU units (chlu.core.lattice.CLULattice):
    position-only coupling on a declared edge list, one global Verlet step,
    designed inertial-mass banding. The experiment measures the F5 §7.2
    communication-pricing claim on a designed 2-unit SO(2) lattice (sync
    timescale ∝ kappa_c^{-1/2}; relative-memory retention ∝ 1/kappa_c; shared
    channel = exact latch at every kappa_c), runs N ∈ scaling_sizes chain
    smoke checks (joint symplecticity, energy drift, wall-clock), a wormhole
    gate smoke, and a small banded-vs-uniform training smoke.

    Nomenclature per F5 Def-2: inertial mass M vs spectral mass mu.
    """

    # --- lattice / coupling knobs (build_lattice) ---
    # "auto" = channel_spring for so2_invariant units (U(1)-preserving; the
    # xy-lattice-theory P5 design rule), "spring" for every other potential
    # type — so this default is behavior-identical to the old "spring" for
    # every lattice this experiment builds (all use potential_type="mlp").
    # Explicit: "spring" (kappa*||W_i q_i - W_j q_j||^2, learnable W),
    # "channel_spring" (fixed identity-on-channel W: exact XY, J = 2 kappa r*^2),
    # or "mlp".
    coupling_type: str = "auto"
    coupling_dim: int = 2
    kappa_c: float = 0.05  # coupling strength for trained/smoke lattices
    proj_init_scale: float = 0.1  # init scale of learnable spring projections W
    # "random" (legacy; a generic W breaks the lattice's global U(1)) or
    # "conformal" (W = 1_k at init, still trainable). Only used by "spring".
    proj_init_mode: str = "random"
    kinetic_energy_mode: str = "newtonian_learned"  # banding needs log_mass read
    hidden_dim: int = 32
    dt: float = 0.05

    # --- pricing measurement (designed 2-unit SO(2) lattice; no training) ---
    hat_lambda: float = 1.0  # Mexican-hat stiffness (radial mu^2 = 8*lam*f^2/M)
    vacuum_radius: float = 1.0  # designed vacuum circle radius f
    channel_inertia: float = 1.0  # designed channel inertial mass M (per unit)
    kappa_sweep: List[float] = field(
        default_factory=lambda: [0.003, 0.01, 0.03, 0.1, 0.3]
    )
    sync_delta0: float = 0.4  # initial relative angle for the sync-time probe
    sync_max_steps: int = 4000  # gamma=0 rollout cap for first alignment
    probe_gamma: float = 0.2  # overdamped probe friction (h* = 0.111)
    probe_kick: float = 0.05  # canonical kick size for retention/latch probes
    max_probe_steps: int = 24000  # retention rollout cap (smallest kappa is slowest)
    latch_steps: int = 3000  # shared-channel latch probe length

    # --- scaling smoke ---
    scaling_sizes: List[int] = field(default_factory=lambda: [2, 4, 8])
    scaling_steps: int = 2000  # energy-drift & wall-clock rollout length

    # --- wormhole skeleton ---
    wormhole_gate_threshold: float = 1.0  # gate opens for V_c below ~threshold
    wormhole_gate_width: float = 0.25  # smooth gate width (energy units)
    # "free_energy" (default) = -w*softplus((t-v)/w), the annealed free energy of
    # the Ising gate; force = <sigma>*grad v, monotone & attractive. "mean_energy"
    # = the legacy <sigma>*v, whose force reverses sign at v = 0.802 (t=1, w=0.25)
    # — the wormhole repels its own endpoints. Legacy is kept, never defaulted.
    gate_energy_mode: str = "free_energy"

    # --- training smoke (2-unit banded vs uniform, single seed = indicative) ---
    train_epochs: int = 300
    use_pretrained: bool = False
    train_n_traj: int = 64
    train_seq_len: int = 256
    train_window: int = 64
    data_omegas: List[float] = field(default_factory=lambda: [0.5, 2.0])
    data_masses: List[float] = field(default_factory=lambda: [4.0, 0.25])
    data_radius: float = 1.0
    banded_mass_scales: List[float] = field(default_factory=lambda: [4.0, 0.25])
    train_kappa_c: float = 0.01  # small learnable coupling during training smoke
    eval_steps: int = 255  # held-out rollout length for eval MSE


@dataclass
class ExperimentPaidAccessConfig:
    """Configuration for the w7 paid-access battery (V1 pillar-4 gate).

    The discriminating end-to-end test of intra-unit *access* mechanisms
    (paid-access-theory §7.1-7.3): does a relativistic CLU's causal box
    C_T (Prop-A2, half-width L_i = T*eps*c/sqrt(M_i)) gate reach exactly as
    predicted, and do the paid mechanisms (squeeze cures escape, wormhole
    cures reach) hold their certificates?

    §7.1 multi-basin REACH task: a K-basin analytic potential with basin
    distances d_k spanning below AND above L; arms = plain relax, S^(M)
    squeeze (line-searched zeta), intra-unit wormhole (matched channel),
    Newtonian-squeeze control (energy DOES buy reach), no-physics router
    (CM-7), dense/throat-V discriminator. §7.2 latch transit. §7.3
    certificates on every arm.

    Mass banding is a PREREQUISITE (theory §3.3 reason 2): all squeeze arms
    run with a designed band so S^(M) is directional (else the l0-gate
    uniform-M ambiguity repeats). ``mass_band`` states it.
    """

    # --- geometry / physics ---
    dim: int = 2  # small (2-4); channel = coords (0, 1)
    c: float = 1.0  # speed of causality (relativistic cap c/sqrt(M))
    rest_mass: float = 1.0  # m0
    dt: float = 0.05  # Verlet step
    reach_steps: int = 100  # T: rollout horizon that sets the causal box L
    # (default band [4.0,0.25] => M0=4 => L = T*dt*c/sqrt(M0) = 100*0.05/2 = 2.5,
    #  which sits mid-list in basin_distances so some d<L, some d>L.)
    # Designed inertial-mass band applied to log_mass (softplus). The reach
    # direction is coord 0; a LIGHT coord-0 mass gives a wide box, heavy the
    # opposite — band = [M_light_dir_scale, M_other...] via softplus target.
    mass_band: List[float] = field(default_factory=lambda: [4.0, 0.25])

    # --- K-basin potential (analytic double/triple well along coord 0) ---
    barrier_height: float = 1.0  # Delta V_b between adjacent basins
    basin_curvature: float = 4.0  # quadratic stiffness inside each basin
    # basin center distances d_k from the start basin, along coord 0. The
    # experiment places basins at 0, +d_1, +d_2, ... and reports L so some
    # d_k < L (reach OK) and some d_k > L (reach FAILS).
    basin_distances: List[float] = field(
        default_factory=lambda: [0.8, 1.6, 2.4, 3.2, 4.0, 5.0]
    )

    # --- governor / relaxation (fixed so plain relax provably cannot escape) ---
    gamma: float = 0.1  # dissipation for plain relaxation
    relax_steps: int = 400  # relaxation length after any injection
    init_momentum: float = 1.2  # p0 along coord 0 (KE0 < Delta V_b by design)

    # --- squeeze (line search over rapidity) ---
    zeta_grid: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
    )

    # --- wormhole channel (oracle placement; radii) ---
    capture_radius: float = 0.35

    # --- throat / dense-V discriminator ---
    throat_depth: float = 1.5  # lowers the barrier between adjacent basins

    # --- latch transit (§7.2) ---
    latch_radius: float = 3.0  # vacuum-circle radius f for SO(2) sector
    latch_momentum: float = 0.5  # p scale for the charge Q = p^T X q

    # --- certificate payoff A: router latch erasure (§7.2b, referee F3.1) ---
    # A cloud of incoming states inside the capture ball is pushed through each
    # arm. The wormhole (canonical translation, det J = 1) is injective: it
    # TRANSPORTS the spread of Q by the exact constant p^T X Delta. The
    # no-physics router (q := exit, det J = 0) is non-injective: every incoming
    # state exits with the same Q => the latch is ERASED (Var(Q_out) = 0).
    payoff_latch_samples: int = 16  # incoming states drawn in the capture ball
    payoff_capture_jitter: float = 0.3  # <= capture_radius; spread of incoming q

    # --- certificate payoff B: coercive-exit BIBO (§7.4, theory issue 7) ---
    # V(q) = 0.5*k*q0^2 - eps*q0^4 (+ transverse confinement): coercive only
    # inside the component |q0| < x_b = sqrt(k/(4 eps)); barrier V_b =
    # k^2/(16 eps). Exits are requested BOTH inside and outside that component.
    # NOTE (why an energy-only test is not enough): V(4.0) < V_b even though
    # 4.0 > x_b -- the receipt must test COMPONENT MEMBERSHIP, not just energy.
    bibo_k: float = 1.0  # coercive curvature near the origin
    bibo_quartic: float = 0.02  # -eps*q0^4 non-coercive tail => x_b = 3.536
    bibo_exit_distances: List[float] = field(
        default_factory=lambda: [1.0, 2.0, 3.0, 3.6, 4.0, 5.0]
    )
    bibo_init_momentum: float = 0.3  # p0 along coord 0 at the entrance
    bibo_steps: int = 2000  # T; r* measured at T and 2T (growth test)
    bibo_gamma: float = 0.02  # dissipation: bounded arms must settle
    bibo_escape_radius: float = 20.0  # ||q|| beyond this counts as escaped
    bibo_margin: float = 1e-3  # receipt safety margin on both tests

    # --- landing criterion / seeds ---
    landing_tol: float = 0.4  # |q0 - basin_center| < tol counts as landed
    n_seeds: int = 5
    seed0: int = 0
    quick_seeds: int = 2
    quick_distances: int = 3  # first N basin_distances in --quick


@dataclass
class ExperimentS1Config:
    """Configuration for the S1 pilot: trash-region Pareto (gamma-field study).

    Signal attractor (Figure-8 lemniscate, Exp-A machinery) + structured noise
    injected from a localized off-attractor cluster. Arms: (i) global gamma
    sweep, (ii) energy governor, (iii) learned gamma_phi (K in
    ``learned_k_values``), (iv) fixed oracle hole at the known noise locus.
    Metric: signal-retention vs noise-rejection Pareto (brainstorm Thread 1;
    F5 Def-5/Prop-11; C1 comparison per mo-deep-read §5).
    """

    dt: float = 0.05
    n_train_cycles: int = 3
    window_size: int = 64
    train_epochs: int = 500
    hidden_dim: int = 64
    kinetic_energy_mode: str = "newtonian_identity"  # match Exp-A defaults
    use_pretrained: bool = False
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    sleep_steps: int = 100  # shorter sleep evolution than the global default

    # Structured noise = the garbage source: a Gaussian cluster at a fixed
    # off-attractor locus (known, so arm (iv) can place the oracle hole on it).
    # Kept >~1.5 units off the lemniscate: sigmoid horizon tails leak friction
    # onto the curve if the locus is closer (observed in the smoke run at
    # [1.2, 1.2] — the oracle's own tail damped the signal orbit).
    noise_center: List[float] = field(default_factory=lambda: [1.5, 1.5])
    noise_q_std: float = 0.15
    noise_p_std: float = 0.6
    # Fraction of the replay buffer seeded at the noise locus: the training
    # environment EXPOSES the garbage source to the sleep phase; the field
    # must still LEARN to place friction there (exposure != placement).
    buffer_noise_frac: float = 0.3

    # Evaluation protocol
    eval_clean_steps: int = 2000  # retention horizon (clean free-run)
    eval_kick_steps: int = 400  # rejection horizon per injection
    n_injections: int = 16
    # Arm (i): global-gamma sweep traces the Pareto trade-off curve
    # (0.0 = conservative reference point / model ceiling)
    global_gamma_sweep: List[float] = field(
        default_factory=lambda: [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    )
    governor_sensitivity: float = 0.95  # arm (ii)

    # Arm (iii): learned fields (one trained model per K per seed). Field
    # geometry (gamma_max, width, init_*) comes from training.friction_field_*
    # — single source of truth.
    learned_k_values: List[int] = field(default_factory=lambda: [1, 4])
    # Arm (iv): oracle hole hand-placed at noise_center. Width is harder than
    # the learned default: the frozen control cannot retreat from the curve,
    # so its horizon tail must not reach it.
    oracle_radius: float = 0.6
    oracle_strength: float = 0.3
    oracle_width: float = 0.1


@dataclass
class ExperimentKTConfig:
    """Configuration for the Kosterlitz-Thouless memory-phase suite (Thread-10).

    An ``L x L`` torus of designed SO(2) CLU registers (``channel_spring(kappa)``
    + ``MexicanHatPotential``, ``newtonian_learned``, no governor) reduces on its
    vacuum ring to the 2-D XY model with ``J = 2 kappa r*^2``, giving the
    Nelson-Kosterlitz universal jump at

        T_KT = 1.786 kappa r*^2 = **0.0893** CLU units at kappa=0.05 (= 0.8929 J)

    ⚠ NOT "0.1786" — that value is wrong by a factor 2 and is retracted.

    Defaults reproduce the validated ``kt-2d-csf3`` laptop run exactly (that is
    the round-trip acceptance gate, ``tests/test_kt.py``). The CSF3/A100 tranche
    overrides them explicitly on the sbatch command line — see
    ``scripts/csf3/job_gpu_kt.sh`` — so every scaled run carries its own
    provenance. See chlu/experiments/kt/.
    """

    # ---- designed-register physics (the XY dictionary J = 2 kappa r*^2) ----
    lam: float = 1.0  # Mexican-hat quartic; k_r = 8 lam f^2 = 8
    f: float = 1.0  # vacuum radius parameter (r* = f = 1)
    kappa: float = 0.05  # channel spring; kappa/k_r = 0.00625 (Born-Oppenheimer safe)
    rstar: float = 1.0
    dt: float = 0.02  # Langevin step
    gamma: float = 0.10  # Langevin friction
    n_walkers: int = 256  # vmapped walkers (the A100 knob: 1024+ is cheap)
    kinetic_mode: str = "newtonian_learned"
    # ⚠ handover §7.22: the repo default is "legacy", under which T is NOT in
    # energy units and NONE of this physics holds. clu_path.assert_kt_settings
    # raises if this is not "fdt".
    langevin_noise: str = "fdt"

    # ---- mode winding1d: 1-D CLU ring, bias-free winding MSD (exponent (b)) ----
    # Laptop: T/J=1.0 gave slope -0.7 because xi~1.2 there (slips not
    # independent). The clean tau ~ 1/N slope -1 needs T/J=0.5 (xi~2.8) + long
    # runs: that is the CSF3 override.
    winding1d_n_values: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    winding1d_tj: float = 1.0
    winding1d_chunks: int = 300  # N <= 16
    winding1d_chunks_large: int = 200  # N > 16
    winding1d_chunk_steps: int = 100
    winding1d_seed: int = 31
    # Fit the MSD only over the diffusive window MSD <= this value. None keeps
    # the original full-range fit (bit-exact round-trip vs the laptop JSONs) —
    # but see clu_path.run_winding_msd: the full-range fit is SATURATION-
    # DOMINATED at T/J=1.0, which is very likely why the laptop slope came out
    # -0.7. Set ~0.3 for any run whose purpose is the exponent.
    winding1d_msd_fit_max: Optional[float] = None

    # ---- mode winding2d: 2-D winding survival tau(L) (exponent (a)) ----
    # Laptop reached L<=16 only, where vortex-diffusion traversal (~L^2) masks
    # the negative Arrhenius exponent above T_KT. L>=32 is the CSF3 override.
    winding2d_l_values: List[int] = field(default_factory=lambda: [8, 12, 16])
    winding2d_tj_values: List[float] = field(
        default_factory=lambda: [0.60, 0.70, 1.10, 1.30]
    )
    winding2d_nwalk: int = 24
    winding2d_nmax_below: int = 20000  # first-passage censor below T_KT
    winding2d_nmax_above: int = 6000  # ... and above
    winding2d_tkt_over_j: float = 0.9  # which censor applies (measured T_KT/J = 0.898)
    winding2d_seed: int = 700

    # ---- mode bridge: CLU-Langevin vs reduced-XY rho_s (kill criterion) ----
    bridge_l: int = 8
    bridge_tj_values: List[float] = field(default_factory=lambda: [0.70, 0.85, 1.00])
    bridge_chunks: int = 40
    bridge_burn_chunks: int = 10
    bridge_chunk_steps: int = 200
    bridge_seed: int = 7
    bridge_equil_seed: int = 1234
    bridge_equil_sweeps: int = 1500

    # ---- mode reduced: reduced-XY phase diagram (sections B, C, F) ----
    reduced_l_values: List[int] = field(default_factory=lambda: [8, 16, 32])
    reduced_tj_values: List[float] = field(
        default_factory=lambda: [
            0.50,
            0.60,
            0.70,
            0.80,
            0.85,
            0.90,
            0.95,
            1.00,
            1.10,
            1.20,
        ]
    )
    reduced_seeds: List[int] = field(default_factory=lambda: [100, 101, 102])
    reduced_nwalk_small: int = 4  # L <= 16
    reduced_nwalk_large: int = 2  # L > 16
    reduced_therm_small: int = 1500
    reduced_therm_large: int = 3000
    reduced_meas_small: int = 4000
    reduced_meas_large: int = 6000
    reduced_meas_every: int = 5
    # section C: twist-response route B (⚠ leaks the w=0 sector at L>=16 near T_KT)
    reduced_twist_l_values: List[int] = field(default_factory=lambda: [8, 16])
    reduced_twist_tj_values: List[float] = field(
        default_factory=lambda: [0.60, 0.80, 0.90, 1.00, 1.10]
    )
    reduced_twist_a: float = 0.2
    # section F: broken-symmetry null (the random-W p=2 anisotropy) -> no KT jump
    reduced_broken_l: int = 16
    reduced_broken_h2: float = 1.0
    reduced_broken_tj_values: List[float] = field(
        default_factory=lambda: [0.60, 0.80, 0.90, 1.00, 1.20]
    )


@dataclass
class ExperimentMinusPhysicsConfig:
    """Configuration for the 'CLU minus the physics' controls (G2 / P6).

    Three identical-capacity arms — CHLU (symplectic), BrokenVolumeCHLU
    (det J != 1), UnconstrainedTwin (free residual recurrence) — run through
    ONE measurement protocol on the SO(2)-degenerate circle-vacuum task,
    isolating what symplecticity (integrator structure | volume conservation)
    functionally buys. See chlu/experiments/exp_minus_physics.py.
    """

    dim: int = 4
    hidden_dim: int = 64
    # "mlp" keeps the potential architecture IDENTICAL between CHLU and the
    # broken-volume arm, so the ablation isolates symplecticity (not a designed
    # SO(2) potential). Emergent flat direction, if any, comes from the data.
    potential_type: str = "mlp"
    kinetic_energy_mode: str = "newtonian_learned"
    dt: float = 0.05
    train_epochs: int = 150
    n_seeds: int = 3
    measure_erosion: bool = True

    # Dataset (circle-vacuum, shared with Experiment D)
    n_points: int = 256
    seq_len: int = 65
    circle_radius: float = 1.0

    # Measurement harness
    settle_gamma: float = 0.1
    settle_steps: int = 2000
    # BIBO diagnostic: settled radius above settle_bound * R counts as diverged
    # (the volume-breaking arms lose the bounded attractor; F5 Prop-10 / §7.7).
    settle_bound: float = 20.0
    probe_gamma: float = 0.05
    probe_steps: int = 4000
    probe_kick: float = 0.1
    eval_steps: int = 400


@dataclass
class ExperimentRetrievalConfig:
    """Configuration for the hand-built write -> address -> retrieve loop.

    Stage-1 empirical test of the Head's addressable-dynamical-memory vision
    (handover 2026-07-21): a HAND-DESIGNED ``RingRegisterPotential`` holding K
    items, hand-picked addresses ``(m, q0, p0)``, and a LINEAR read on the
    rollout tail. **Nothing here is learned** except the address itself in the
    item-5 restructuring test. See chlu/experiments/exp_retrieval.py.
    """

    # ---- designed landscape geometry ----
    lam: float = 1.0  # ring quartic; radial spectral mass mu_rad^2 = 8*lam*f^2
    f: float = 1.0  # vacuum radius
    # Angular barrier between item sites. 0.2, NOT 0.05: at 0.05 the well is
    # too weak to hold a JITTERED query within the rollout (measured: mean
    # payload error 0.18 at K=8, vs 0.002 at 0.2) — the particle never settles
    # and every downstream number degrades for a reason that has nothing to do
    # with memory capacity.
    barrier: float = 0.2
    payload_kappa: float = 1.0  # payload-channel spring constant
    bump_width: float = 0.05  # payload bump width (FIXED as K grows -> interference)
    payload_seed: int = 0  # seed for the designed non-monotone payload values

    # ---- rollout ----
    dt: float = 0.05
    gamma: float = 0.02  # friction: needed for the particle to SETTLE in a well
    steps: int = 1200
    tail_frac: float = 0.25  # read only the tail (the head still carries the address)
    n_subsample: int = 8

    # ---- queries / linear probe ----
    n_query_per_item: int = 64
    query_sigma_theta: float = 0.15
    query_sigma_r: float = 0.05
    query_sigma_p: float = 0.05
    payload_tol: float = 0.1  # |readout - a_k| below this counts as "settled"
    # Where along the rollout the survival read is taken (fraction of `steps`)
    survival_fracs: List[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.25, 0.5, 0.75, 0.99]
    )

    # ---- item 2: mass as address key ----
    mass_probe_K: int = 8
    mass_probe_p: float = 0.5  # fixed |p0| components at the launch point
    mass_log_lo: float = -1.5
    mass_log_hi: float = 1.5
    mass_n: int = 25  # scalar sweep resolution
    mass_n_vec: int = 9  # per-axis resolution of the (m0, m1) grid
    # Robustness of the mass key: a key is only usable if a small error in the
    # mass still retrieves the same item. Counting DISTINCT items reachable by
    # mass is not enough — a chaotic map has many reachable items and is still
    # useless as an address.
    mass_jitter_rel: float = 0.01
    mass_jitter_n: int = 8
    mass_robust_threshold: float = 0.9  # frac_same needed to count a mass cell

    # ---- item 3: three write modes (ThreeModePotential, dim=4) ----
    tm_beta: float = 1.0
    tm_d: float = 1.0
    tm_write_theta: float = 0.7
    tm_write_dr: float = 0.3
    tm_write_sign: int = 1
    tm_steps: int = 2000

    # ---- item 4: interference ----
    item_counts: List[int] = field(default_factory=lambda: [2, 4, 8, 16, 32])
    selectivity_threshold: float = 0.9

    # ---- item 5: address restructuring (the learnability crux, weak form) ----
    restructure_K: int = 8
    restructure_offsets: List[int] = field(default_factory=lambda: [1, 2, 4])
    address_lr: float = 0.05
    address_steps: int = 300
    smooth_n_theta: int = 181
    smooth_rollout_steps: List[int] = field(
        default_factory=lambda: [25, 50, 100, 200, 400, 800]
    )
    # gamma scan for the retrieval-vs-learnability tension (friction is what
    # makes the read stable AND what kills the address gradient)
    smooth_gammas: List[float] = field(
        default_factory=lambda: [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    )
    # item 5c: gamma-annealed address search (the repair implied by 5b)
    anneal_gamma_lo: float = 0.001
    anneal_n_stages: int = 6


@dataclass
class ExperimentDimScalingConfig:
    """Configuration for the address-space DIMENSION-SCALING measurement (w20).

    w19 (``exp_retrieval``) measured a capacity ceiling of **8 items** on a 2-D
    ring. That 8 is a property of the ring's *angular* resolution
    (``K_max ~ 0.2 * 2*pi / sigma_theta``), not of CLU — the theorist's packing
    bound ``(1 + 2R/w)^d`` is exponential in the address dimension ``d``. This
    experiment generalizes the ring to a ``d``-dimensional address ball
    (``BallRegisterPotential``) and measures ``K_max`` vs ``d``.

    Fidelity criteria are the w19 criteria **verbatim** so the numbers are
    comparable, including the mandatory blank-landscape control on every cell.
    See chlu/experiments/exp_dim_scaling.py.
    """

    # ---- designed landscape geometry ----
    R: float = 1.0  # radius of the region the SITES occupy (the "R" of the bound)
    # Clearance between the outermost sites and the confining wall. Load-bearing,
    # NOT cosmetic: farthest-point sampling pushes sites onto the boundary of the
    # site region, so with zero margin a query jittered OUTWARD from a boundary
    # site starts outside the wall. Measured at d=2: such a query began at
    # V=+0.68 with KE=0.44, was slingshotted across the ball and captured by a
    # well 1.37 away -- 4/128 queries, capping selectivity at 0.94 even at K=2.
    # 0.5 > 3 * query_sigma keeps 3-sigma queries inside the force-free region.
    wall_margin: float = 0.5
    well_width: float = 0.15  # Gaussian well width w (the "w" of the packing bound)
    well_depth: float = 1.0  # b: well depth
    # Payload-channel spring constant. The payload term 0.5*kappa*(y - s(x))^2
    # exerts a force kappa*(y-s)*s'(x) ON THE ADDRESS PLANE, so kappa trades the
    # read-out's settling speed against how much it perturbs addressing.
    # Measured at the densest resolvable d=2 cell (K=16), codebook read accuracy:
    #   kappa   1.00    0.30    0.10    0.03
    #   acc     0.508   0.898   0.906   0.766
    # kappa=1 is too stiff (the payload force perturbs the address once wells are
    # close); kappa=0.03 is too slack (payload has not settled within the
    # rollout: |err| 3e-3 vs 4e-5). 0.1 is the measured optimum.
    # NOTE: selectivity is essentially kappa-independent (0.930-0.945 across the
    # whole sweep) -- kappa affects the READ, not the addressing. The query
    # ejection that capped selectivity at K=2 was `wall_margin`, not kappa.
    payload_kappa: float = 0.1
    c_conf: float = 10.0  # stiffness of the confining wall outside the ball
    site_seed: int = 0  # seed for the farthest-point site packing
    payload_seed: int = 0  # seed for the designed non-monotone payload values

    # ---- rollout (w19 values, so the cells are comparable) ----
    dt: float = 0.05
    gamma: float = 0.02  # friction: needed for the particle to SETTLE in a well
    steps: int = 1200
    tail_frac: float = 0.25
    n_subsample: int = 8
    # Rollouts are vmapped in chunks: a (K * n_query) x steps x 2*dim trajectory
    # buffer is what actually bites at large K, not the flop count.
    rollout_chunk: int = 512

    # ---- queries / linear read (w19 criteria verbatim) ----
    n_query_per_item: int = 32
    # Total-query budget per cell. Cost scales as n_queries * steps * K (the
    # potential sums over all K wells at every step), so the per-item count is
    # reduced at large K to keep a cell affordable:
    #   n_eff = clip(max_total_queries // K, min_query_per_item, n_query_per_item)
    # steps is deliberately NOT reduced instead: at steps=600 the borderline
    # d=2 K=16 cell flips from 0.906 to 0.867 accuracy, which would move K_max.
    # The 1200-step rollout is load-bearing for comparability with w19.
    max_total_queries: int = 8192
    min_query_per_item: int = 4
    query_sigma: float = 0.15  # address jitter magnitude (w19 sigma_theta = 0.15)
    # How query_sigma is interpreted as d grows. "fixed_norm" (default) sets the
    # per-axis scale to sigma/sqrt(d) so the jitter NORM is sigma at every d --
    # the apples-to-apples generalization of w19's 1-D ring arc jitter, holding
    # query PRECISION fixed and varying only the address dimension. "per_axis"
    # uses sigma literally, so the jitter norm grows as sigma*sqrt(d). The two
    # give materially different capacity curves; both are reported.
    query_noise_mode: str = "fixed_norm"
    query_sigma_p: float = 0.05
    payload_tol: float = 0.1
    selectivity_threshold: float = 0.9  # w19 threshold for "retrieved"
    # A cell whose BLANK control beats chance by more than this is discarded, not
    # reported: it means the read is leaking the address, not reading the memory.
    blank_margin: float = 0.15

    # ---- item 1: the K_max vs d curve ----
    # d=1 is included deliberately: w19's ring is a 1-D address MANIFOLD embedded
    # in 2-D, so the ring ceiling's comparison point is d=1, not d=2.
    dims: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 6, 8, 12, 16])
    # Geometric K ladder; the search stops at the first K below threshold.
    k_ladder: List[int] = field(
        default_factory=lambda: [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    )
    k_cap: int = 2048  # compute cap; a cell stopped here is reported CENSORED

    # ---- capture radius (the MEASURED w of the packing bound) ----
    capture_n_dirs: int = 16
    capture_n_offsets: int = 24
    capture_max_offset: float = 1.0

    # ---- item 2: basin-width sweep at fixed d ----
    width_sweep_dims: List[int] = field(default_factory=lambda: [2, 3, 4])
    width_sweep: List[float] = field(
        default_factory=lambda: [0.08, 0.15, 0.22, 0.30, 0.45]
    )
    # The width sweep is 3 dims x 5 widths = 15 ladders, so it gets a tighter
    # compute cap than the headline curve. Cells stopped here are CENSORED and
    # the sweep is read for its SHAPE (plateau vs power law), not absolute K_max.
    width_sweep_k_cap: int = 512

    # ---- item 5: ADDRESSING capacity, decoder-free ----
    # Item 1 walks the ladder on the w19 codebook read, which stops when the
    # single scalar payload channel runs out of resolution -- at d >= 6 that
    # happens while addressing is still perfect (d=6, K=512: read 0.599,
    # selectivity 1.000). This sweep re-walks the ladder on selectivity alone so

    # ---- item 4: does dissipation still gate retrieval at d > 2? ----
    gamma_sweep_dims: List[int] = field(default_factory=lambda: [2, 4, 8])
    gamma_sweep: List[float] = field(
        default_factory=lambda: [0.0, 0.002, 0.005, 0.01, 0.02, 0.05]
    )
    gamma_sweep_K: int = 8
@dataclass
class ExperimentLearnedMemoryConfig:
    """Configuration for the LEARNED write -> address -> read loop (w20).

    w19 (``ExperimentRetrievalConfig``) ran the loop on a HAND-DESIGNED
    landscape. This group runs the same loop on a landscape whose potential is
    **trained** (``chlu.training.train_memory``), across a design-freedom ladder
    from the w19 hand-built landscape to a free MLP, and measures the
    fidelity-vs-design-freedom curve. See chlu/experiments/exp_learned_memory.py.

    Geometry defaults are deliberately IDENTICAL to the w19 group so the two
    experiments are directly comparable; only the potential family changes.
    """

    # ---- geometry (matched to ExperimentRetrievalConfig) ----
    lam: float = 1.0
    f: float = 1.0
    barrier: float = 0.2
    payload_kappa: float = 1.0
    bump_width: float = 0.05
    payload_seed: int = 0

    # ---- design-freedom ladder ----
    rungs: List[str] = field(
        default_factory=lambda: [
            "designed",
            "skeleton_residual",
            "sites_learned_payload",
            "local_rbf",
            "free_mlp",
        ]
    )
    hidden: int = 64  # learned MLP width
    n_atoms: int = 24  # learned RBF dictionary size
    residual_scale: float = 0.1  # learned residual weight at rung `skeleton_residual`
    rbf_init_width: float = 0.3

    # ---- write objective (chlu/training/train_memory.py) ----
    write_steps: int = 600
    write_lr: float = 3e-3
    write_weight_decay: float = 1e-4
    write_n_perturb: int = 32
    write_sigma_addr: float = 0.25
    write_sigma_pay: float = 0.6
    write_margin: float = 0.15
    write_barrier: float = 0.2

    # ---- TWO-PHASE retrieval ----
    # Phase 1 relaxes the query to its address (dissipative, no gradient needed);
    # phase 2 rolls out from the address for the read (where gradients are safe).
    dt: float = 0.05
    gamma_address: float = 0.05
    gamma_read: float = 0.0
    address_steps: int = 400
    read_steps: int = 800
    tail_frac: float = 0.25
    n_subsample: int = 8

    # ---- queries / read ----
    n_query_per_item: int = 32
    query_sigma_theta: float = 0.15  # x f => q-space jitter, matched to w19 arc length
    query_sigma_p: float = 0.05
    payload_tol: float = 0.1
    item_counts: List[int] = field(default_factory=lambda: [2, 4, 8, 16])
    survival_fracs: List[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.25, 0.5, 0.75, 0.99]
    )

    # ---- pass criteria for the minimum-viable-design point ----
    # The design-freedom question is only asked at item counts where the
    # reference (fully designed) landscape itself passes — otherwise a rung is
    # charged for the ring's CAPACITY ceiling (w19: ~8 items) rather than for
    # anything to do with learning.
    reference_rung: str = "designed"
    pass_strict: float = 0.9
    pass_read: float = 0.9
    # A blank landscape must read within this of chance, else the cell is not a
    # measurement (w19: blank 0.469 vs chance 0.500).
    blank_margin: float = 0.15
    # A blank landscape must also fail the VALUE read: it stores 0 everywhere, so
    # its strict-success rate must be ~0. This control is leak-immune, which is
    # why it, not the classification control, defines the primary criterion.
    blank_strict_max: float = 0.1

    # ---- item 3: cross-write interference ----
    interference_K: int = 4
    interference_write_steps: int = 300

    # ---- item 4: the 2-D gamma map ----
    gamma_map_K: int = 4
    gamma_map_rungs: List[str] = field(
        default_factory=lambda: ["designed", "sites_learned_payload"]
    )
    gamma_address_grid: List[float] = field(
        default_factory=lambda: [0.0, 0.005, 0.02, 0.05, 0.1]
    )
    gamma_read_grid: List[float] = field(
        default_factory=lambda: [0.0, 0.005, 0.02, 0.05, 0.1]
    )

    # w19 reference numbers, carried in the results JSON so any comparison in a
    # report is made against the actual measured baseline, not a remembered one.
    w19_baseline: Dict[str, float] = field(
        default_factory=lambda: {
            "payload_abs_err": 9.98e-4,
            "codebook_read_K2": 1.000,
            "codebook_read_K8": 0.992,
            "blank_control_K2": 0.469,
            "designed_corruption": 4.17e-7,
        }
    )


@dataclass
class ExperimentPotentialClassConfig:
    """Configuration for the POTENTIAL FUNCTION CLASS sweep (w21, Task D).

    w20 found that no learned rung of the design-freedom ladder clears strict
    0.9 at both K=4 and K=8, that write locality collapses (0.000 -> 2.9e-2 ..
    5.0e-1) and that one subsequent write destroys the best rung. **Every w20
    learned rung used ``PotentialMLP``.** Two hypotheses explain that:

    * **H-EXPR** — the MLP is too weak; more capacity fixes it.
    * **H-SUPP** — the failure is *global parameter support*: any weight update
      moves every stored item, so attention (more global still) should fail at
      least as badly and only *atom* writes can be local.

    They make OPPOSITE predictions on the transformer arm, which is why this
    group sweeps the **function class** with everything else held at the w20
    values (the geometry/retrieval fields below are deliberately identical to
    ``ExperimentLearnedMemoryConfig`` so the two experiments are comparable).
    See chlu/experiments/exp_potential_class.py.
    """

    # ---- geometry / retrieval: IDENTICAL to the w20 group ----
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
    survival_fracs: List[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.25, 0.5, 0.75, 0.99]
    )
    blank_margin: float = 0.15
    blank_strict_max: float = 0.1
    pass_strict: float = 0.9
    pass_read: float = 0.9
    reference_rung: str = "designed"

    # ---- write objective: IDENTICAL to the w20 group ----
    write_steps: int = 600
    write_lr: float = 3e-3
    write_weight_decay: float = 1e-4
    write_n_perturb: int = 32
    write_sigma_addr: float = 0.25
    write_sigma_pay: float = 0.6
    write_margin: float = 0.15
    write_barrier: float = 0.2
    # A LOCAL write is one masked, single-item write per item. Same per-write
    # step budget as the global write; because each sub-write carries one target
    # instead of K, total write FLOPs stay comparable (reported in the cost
    # table, never assumed).
    local_write_steps: int = 600

    # ---- ⭐ the swept variable: the potential's function class ----
    # "designed"       rung 0, zero learned parameters (the ceiling)
    # "mlp"            w20 baseline (PotentialMLP)                 GLOBAL support
    # "hopfield"       modern-Hopfield energy at beta_soft         support ~exp(-beta d)
    # "hopfield_sharp" the same class at beta_sharp                (capacity held FIXED)
    # "attn"           cross-attention w/ learned proj + values    GLOBAL-ish
    # "atoms"          Gaussian atom dictionary, GLOBAL grad write LOCAL support
    # "atoms_local"    the same class, per-item MASKED write       LOCAL support + LOCAL write
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

    # ---- matched parameter count (⚠ an unmatched comparison settles nothing) ----
    # PotentialMLP(dim=3, hidden=64) = 3*64+64 + 64*64+64 + 64+1 = 4481.
    param_target: int = 4481
    param_tol: float = 0.05
    hidden: int = 64  # mlp
    n_atoms: int = 896  # atoms: 896 * (3 + 1 + 1) = 4480
    hopfield_n_mem: int = 1120  # hopfield: 1120 * (3 + 1) = 4480
    attn_n_mem: int = 495  # attn: 8*3 + 495*8 + 495 = 4479
    attn_d_head: int = 8

    # ---- family hyperparameters ----
    hopfield_beta_soft: float = 2.0
    hopfield_beta_sharp: float = 8.0
    # alpha = 1/2 makes V EXACTLY the modern-Hopfield energy: grad V = 0 reads
    # q = sum_i softmax(beta <q,k_i>)_i k_i, i.e. one step of attention.
    hopfield_confine: float = 0.5
    hopfield_key_init: float = 0.1
    attn_beta: float = 1.0
    attn_key_init: float = 0.5
    # Coercivity for every family EXCEPT hopfield (which needs alpha=1/2 to be
    # the exact modern-Hopfield energy). 0.05 is PotentialMLP's own confinement,
    # so the arms are coercive on identical terms (F5 Prop-10).
    learned_confine: float = 0.05
    atom_init_width: float = 0.3
    atom_init_scale: float = 1.0
    # Initial atom DEPTH (A_j = amp_j^2, so amp is init'd at sqrt of this). 1e-4
    # => the dictionary starts FLAT and the writer digs the wells. w20's RBFAtoms
    # started at softplus(N(0,0.1)) ~ 0.69 per atom, i.e. rugged on the same
    # length scale retrieval has to traverse - a candidate explanation for the
    # w20 local_rbf failure, measured here rather than assumed. See
    # AtomDictionaryPotential's docstring for why the amplitude is NOT a softplus
    # (a flat softplus start has a 3.4e-4 gradient and the write silently no-ops).
    atom_depth_init: float = 1e-4

    # ---- the sweep ----
    class_item_counts: List[int] = field(default_factory=lambda: [4, 8])
    class_seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])

    # ---- cross-write interference (the H-EXPR/H-SUPP discriminator) ----
    interference_K: int = 4
    interference_write_steps: int = 300
    # 7, not 5: the pre-registered adversarial check ("if an attention arm beats
    # the atom dictionary on interference, re-run at 2 extra seeds") triggered.
    interference_seeds: List[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 4, 5, 6]
    )

    # ---- support radius, MEASURED (decay of ||grad dV|| vs distance) ----
    support_radii: List[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    )
    support_probes_per_radius: int = 96
    support_decay_threshold: float = 0.1  # defines r_10
    support_seeds: List[int] = field(default_factory=lambda: [0, 1, 2])

    # ---- does the design-freedom curve move? (w20 re-run with the best class) ----
    run_ladder_rerun: bool = True
    # Families to re-run the ladder with. "auto" => the best class from the sweep.
    ladder_families: List[str] = field(default_factory=lambda: ["atoms", "hopfield"])
    ladder_rungs: List[str] = field(
        default_factory=lambda: [
            "skeleton_residual",
            "sites_learned_payload",
            "free_mlp",
        ]
    )
    ladder_item_counts: List[int] = field(default_factory=lambda: [4, 8])
    ladder_seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])

    # w20 reference numbers, carried in the results JSON so every comparison is
    # made against the measured baseline rather than a remembered one.
    w20_baseline: Dict[str, float] = field(
        default_factory=lambda: {
            "designed_strict_K4": 1.000,
            "designed_strict_K8": 0.986,
            "free_mlp_strict_K4": 0.853,
            "free_mlp_strict_K8": 0.599,
            "skeleton_residual_strict_K4": 0.903,
            "skeleton_residual_strict_K8": 0.959,
            "local_rbf_strict_K4": 0.623,
            "local_rbf_strict_K8": 0.348,
            "designed_corruption": 0.0,
            "free_mlp_corruption": 3.53e-1,
            "sites_learned_payload_corruption": 4.95e-1,
        }
    )


@dataclass
class ExperimentDesignedMechanismConfig:
    """Configuration for the K=8-wall discriminator (w22): GEOMETRY or LEARNING?

    A fixed designed MECHANISM (``AtomDictionaryPotential``: learned amplitudes/
    centers/widths, group-masked writes) with **learned content** (trained by
    ``chlu.training.train_memory``) on a ``d``-dimensional address ball. Sweeps the
    address dimension ``d`` and measures ``K_learned`` = the largest item count a
    LEARNED atom dictionary clears at strict 0.9 (leak-immune value criterion, blank
    control per cell), overlaid on ``K_designed`` re-measured on the SAME harness
    with a hand-built ``BallRegisterPotential``.

    ⚠ The atom count is scaled with K (``n_atoms = atoms_per_item·K``, ``n_groups =
    K``) so a plateau in ``K_learned`` is a LEARNING failure, not a
    parameterization-capacity failure (theorist §4.3, ``B_total ≤ P·b_θ``). See
    chlu/experiments/exp_designed_mechanism.py.
    """

    # ---- d-ball geometry (shared with the designed dim-scaling arm) ----
    R: float = 1.0
    wall_margin: float = 0.5
    well_width: float = 0.15  # designed BallRegisterPotential well width
    well_depth: float = 1.0
    payload_kappa: float = 0.1
    c_conf: float = 10.0
    site_seed: int = 0
    payload_seed: int = 0

    # ---- two-phase retrieval rollout ----
    dt: float = 0.05
    gamma_address: float = 0.05  # phase 1: dissipative relaxation to the address
    # phase 2: value retrieval REQUIRES dissipation (address-space-dimension-scaling
    # item 4: payload err 0.57 at gamma=0 vs ~1e-6 at gamma=0.02). The payload is the
    # d-th coordinate of the atom-well center, launched at 0, and must dissipate up
    # to a_i — a conservative read (gamma_read=0) leaves it oscillating and the tail
    # mean misses the stored value, so strict success is ~0 even for the DESIGNED arm.
    gamma_read: float = 0.02
    address_steps: int = 400
    read_steps: int = 800
    tail_frac: float = 0.25
    n_subsample: int = 8
    rollout_chunk: int = 256

    # ---- queries + read (fixed_norm jitter, w19-comparable) ----
    n_query_per_item: int = 32
    max_total_queries: int = 4096
    min_query_per_item: int = 4
    query_sigma: float = 0.15
    query_sigma_p: float = 0.05
    payload_tol: float = 0.1
    pass_strict: float = 0.9
    blank_strict_max: float = 0.1  # value blank must score at/below this
    blank_margin: float = 0.15  # classification blank margin over chance (reported)

    # ---- the LEARNED atom-dictionary mechanism ----
    # Atoms PER ITEM. n_atoms = atoms_per_item * K, so the parameter budget scales
    # with K and a plateau is a learning failure, not a capacity-of-parameters one.
    atoms_per_item: int = 32
    # Floor on the total atom count. HARD lower bound, dimension-independent; the
    # dimension-aware geometric floor below (min_atoms_base * min_atoms_c**d)
    # dominates in the discriminator sweep. A large over-complete dictionary smooths
    # the write optimization; scaling atoms DOWN at small K starves the write (d=4
    # K=2 with 64 atoms: write loss stuck at 0.18 on some seeds). The floor keeps
    # every cell over-complete; the atoms_per_item*K term dominates once K is large.
    min_atoms: int = 384
    # ⚠ DIMENSION-AWARE atom floor (w23 dimension-aware-budget). The atom count is
    #     n_atoms = max(atoms_per_item*K, min_atoms, round(min_atoms_base*min_atoms_c**d)).
    # w22 (designed-mechanism-learned-content) scaled the budget with K ONLY, with a
    # FIXED floor (min_atoms=2048 in the clean run). That floor is inadequate at high
    # d: the atoms init N(0, atom_init_scale) in the (d+1)-ball, and the fraction
    # landing near any stored site (radius ~R) DECAYS roughly geometrically per added
    # dimension, so a fixed atom count starves the write at high d (d=8 K=2 stalled at
    # strict 0.400 despite a geometrically-trivial site separation 1.838). The ladder
    # walk then terminates at a starved low-K cell and K_learned reads as an optimizer
    # artifact, not a capacity. A geometric floor c**d compensates: it holds the
    # atoms-near-each-site count ~constant across d. c = min_atoms_c = sqrt(2) is
    # anchored EMPIRICALLY (w23 adequacy probe): the near-site atom count needed for a
    # converged write grows ~sqrt(2) per added dimension — w22 shows d=6 adequate at
    # 2048 atoms and the w23 probe shows d=8 K=16 reaches strict 1.000 by 4096-8192
    # (2048*(sqrt2)^2 = 4096). This is BELOW the a-priori 4*2^d rate (c=2) because the
    # achieved atom packing is non-ideal (address-space-dimension-scaling: d_eff~0.72d,
    # shell concentration), so the effective per-dimension thinning is sub-2. base =
    # min_atoms_base = 512 pins the floor with margin at every sweep dimension (d=2 ->
    # 1024, d=6 -> 4096 = 2x w22, d=8 -> 8192 = strict 1.000 in the probe). Per-point
    # budget adequacy is verified empirically (a failing K must fail with a budget
    # whose further increase does not change the verdict).
    min_atoms_base: int = 512
    min_atoms_c: float = 1.4142135623730951  # sqrt(2)
    # centers ~ N(0, init_scale). ⚠ LOAD-BEARING and MEASURED: at init_scale=0.5 the
    # flat-start atoms cluster near the origin and the writer cannot dig a well that
    # reaches an item whose payload |a_i|=1 from the payload=0 launch manifold — d=2
    # K=4 caps at strict 0.500 (only the |a_i|<1 items retrieve) regardless of atom
    # count (tested to 1024 atoms). At init_scale=1.0 the atoms are spread across the
    # full (d+1)-ball including the payload axis and d=2 K=4 reaches strict 1.000.
    # This is the basin-reach limit potential-function-class flagged (open-Q #2),
    # defused by the initialization, not by more parameters.
    atom_init_scale: float = 1.0
    atom_init_width: float = 0.3  # initial well width s
    atom_depth_init: float = 1e-4  # flat start; the writer digs the wells (A=amp^2)
    learned_confine: float = 0.05  # coercivity alpha*|q|^2
    bits_per_param: int = 32  # for the reported B_total = P * bits_per_param

    # ---- write objective (static, per-item minimum-digging) ----
    write_steps: int = 600  # global write (matched to potential-function-class)
    local_write_steps: int = 300  # per masked single-item write
    write_lr: float = 3e-3
    write_weight_decay: float = 1e-4
    write_n_perturb: int = 32
    write_sigma_addr: float = 0.25
    write_sigma_pay: float = 0.6
    write_margin: float = 0.15
    write_barrier: float = 0.2

    # ---- item 1: the discriminator sweep ----
    dims: List[int] = field(default_factory=lambda: [2, 3, 4, 6, 8])
    k_ladder: List[int] = field(
        default_factory=lambda: [2, 4, 8, 16, 32, 64, 128]
    )
    k_cap: int = 128
    # The discriminator uses the mechanism's BEST single-shot FIDELITY write =
    # GLOBAL (potential-function-class: atoms global 1.000 vs local 0.859 at K=4),
    # so a K_learned plateau cannot be blamed on a suboptimal write operator. The
    # masked-vs-global write is compared explicitly in item 3 (interference).
    learned_arm: str = "learned_global"
    discriminator_seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    designed_seeds: List[int] = field(default_factory=lambda: [0])

    # ---- item 2: mass arm + coupling check ----
    mass_dim: int = 3
    mass_K: int = 4
    mass_spread: float = 4.0  # per-item masses span [1/spread, spread] geometrically
    mass_seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    mass_help_threshold: float = 0.02  # delta-strict above which "mass helps"

    # ---- item 3: interference across d ----
    interference_dims: List[int] = field(default_factory=lambda: [2, 3, 4])
    interference_K: int = 4
    interference_write_steps: int = 300
    interference_seeds: List[int] = field(default_factory=lambda: [0, 1, 2])


@dataclass
class ExperimentPrimitiveHarnessConfig:
    """Configuration for the primitive harness (w20).

    Evaluates the CLU as a GENERAL primitive: MLP / GRU / SSM / attention / CLU
    are dropped into one interchangeable slot (``chlu/core/blocks.py``) at
    matched parameter count, trained identically, on >= 2 task families reported
    SEPARATELY (never averaged). See chlu/experiments/exp_primitive_harness.py.
    """

    # ---- the shared slot (identical for every primitive) ----
    primitives: List[str] = field(
        default_factory=lambda: ["mlp", "gru", "ssm", "attention", "clu"]
    )
    d_model: int = 64
    n_layers: int = 2  # 2, not 1: MQAR needs an induction-head circuit, which a
    # 1-layer attention model provably cannot form (Zoology). A 1-layer harness
    # would hobble the strongest baseline and flatter every other primitive.
    # Budget is matched on the parameters of the BLOCK STACK (the primitive's
    # own parameters, summed over n_layers). Embedding / positional embedding /
    # head are identical across primitives by construction, so including them
    # would dilute the match; total params are reported alongside.
    target_block_params: int = 40000
    param_tol: float = 0.05  # accept |params/target - 1| <= tol
    width_search_lo: int = 4
    width_search_hi: int = 512

    # ---- optimisation (identical for every primitive) ----
    lr_grid: List[float] = field(default_factory=lambda: [3e-4, 1e-3, 3e-3])
    train_steps: int = 2000
    # LR selection runs short (same length for every primitive, so the tuning
    # budget stays equal by construction); the winning LR is then re-trained at
    # full length for n_seeds. 600 steps discriminates the grid cleanly —
    # measured: attention on MQAR T=64 kv=4 reaches 0.44 at lr=1e-3 vs 0.99 at
    # lr=3e-3 by step 800, i.e. the grid ordering is already resolved.
    tune_steps: int = 600
    batch_size: int = 32
    eval_batch: int = 256
    grad_clip: float = 1.0
    n_seeds: int = 3  # seeds for the final (best-LR) numbers

    # ---- family 1: MQAR ----
    mqar_vocab: int = 256
    mqar_seq_lens: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    mqar_kv_fixed: int = 4  # kv held fixed while seq_len sweeps (distractor axis)
    mqar_kv_sweep: List[int] = field(default_factory=lambda: [2, 4, 8, 16])
    mqar_seq_len_fixed: int = 128  # seq_len held fixed while kv sweeps

    # ---- family 2: adding problem ----
    adding_seq_len: int = 128

    # ---- family 3: parity ----
    parity_seq_len: int = 64

    # ---- CLU block physics (defaults, NOT tuned per family) ----
    clu_dt: float = 0.1
    # gamma > 0 was the w20 concession, justified by a w19 measurement that was
    # retracted in the same wave (see CLUBlock docstring). It is now a swept
    # knob; the default is left at the shipped value so w20 numbers reproduce.
    clu_gamma: float = 0.05
    clu_steps: int = 1  # Verlet steps per token
    clu_hidden: int = 32  # potential-MLP hidden width
    clu_kinetic_mode: str = "newtonian_learned"
    clu_potential_type: str = "mlp"
    clu_read_mode: str = "endpoint"  # "endpoint" (settled) | "trajectory" (fiber)
    # EXPLORATORY (w21, outside the pre-registered grid): "linear" is the shipped
    # write current p += W_in x; "gated" multiplies it by sigmoid(W_gate x),
    # supplying the input-conditioned multiplicative write that the GRU, the
    # selective SSM and attention all have. Default preserves shipped behaviour.
    clu_write_mode: str = "linear"
    ssm_selective: bool = True  # Mamba-style input-dependent timescale
    attn_heads: int = 4

    # ---- w21 gamma-read sweep grids (CLU-INTERNAL knobs only) ----
    # Fairness: gamma / clu_steps / read_mode have no counterpart in the MLP,
    # GRU, SSM or attention blocks, so sweeping them is category (a) of the task
    # fairness rule -- a knob no other primitive has. Nothing here touches the
    # shared slot, so the shipped baselines remain directly comparable.
    clu_gamma_sweep: List[float] = field(
        default_factory=lambda: [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    )
    clu_read_mode_sweep: List[str] = field(
        default_factory=lambda: ["endpoint", "trajectory"]
    )
    clu_steps_sweep: List[int] = field(default_factory=lambda: [1, 2, 4])

    # ---- w22 gated-write performance test (exp_gated_write.py) ----
    # The gated-write comparison re-runs the WHOLE five-primitive slot at the
    # published-numbers budget (train 1200 / tune 400) so the re-run baselines
    # double as a reproduction check. These fields drive that experiment only;
    # they do not touch the shipped harness defaults above.
    gw_train_steps: int = 1200
    gw_tune_steps: int = 400
    # MQAR runs the CLU at gamma=0 (gamma-read-sweep §1: +0.040 free, monotone,
    # a category-(a) knob no baseline has); adding/parity keep the shipped gamma.
    gw_mqar_gamma: float = 0.0
    # Item 3a: long-horizon extrapolation — train at gw_extrap_train_T, test at
    # each multiple. The founding CHLU claim (Exp A) is stable extrapolation.
    gw_extrap_train_T: int = 64
    gw_extrap_mults: List[int] = field(default_factory=lambda: [1, 2, 4])
    gw_extrap_families: List[str] = field(
        default_factory=lambda: ["adding", "parity"]
    )
    # Item 3c: robustness — Gaussian noise added to the (continuous) input at
    # inference only. Sweep the std; adding problem (noise on the value channel).
    gw_noise_grid: List[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.1, 0.2, 0.4]
    )


@dataclass
class ExperimentSequentialWriteConfig:
    """Configuration for SEQUENTIAL-WRITE interference + the admission gate (w21).

    w20 measured its worst result — write A, write B, A destroyed (strict
    1.000 -> 0.000) — with **ungated** writes. The theorist measured the same
    contrast with the MVC-0 admission gate and got max drift 8.0e-5. This group
    runs both, on the exact w20 setup and on a K = 1..16 sequential-write curve,
    plus the cross-primitive comparison in the `primitive-harness` slot.
    See chlu/experiments/exp_sequential_write.py.

    Geometry / write / retrieval defaults are deliberately IDENTICAL to
    ``ExperimentLearnedMemoryConfig`` so the gated arm is directly comparable to
    the w20 ungated numbers; only the controller changes.
    """

    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])

    # ---- geometry + landscape family (matched to ExperimentLearnedMemoryConfig) ----
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

    # ---- write objective (chlu/training/train_memory.py) ----
    write_steps: int = 600
    write_lr: float = 3e-3
    write_weight_decay: float = 1e-4
    write_n_perturb: int = 32
    write_sigma_addr: float = 0.25
    write_sigma_pay: float = 0.6
    write_margin: float = 0.15
    write_barrier: float = 0.2

    # ---- TWO-PHASE retrieval (w20 §5: fidelity depends on gamma_address only) ----
    dt: float = 0.05
    gamma_address: float = 0.05
    gamma_read: float = 0.0
    address_steps: int = 400
    read_steps: int = 800
    tail_frac: float = 0.25
    n_subsample: int = 8
    n_query_per_item: int = 32
    n_query_sequential: int = 16  # cheaper: the sequential curve evaluates K times
    query_sigma_theta: float = 0.15
    query_sigma_p: float = 0.05
    payload_tol: float = 0.1
    # ...capped at this fraction of the codebook spacing (w20's ratio at K<=8),
    # so "the stored value came back" stays unambiguous as K grows.
    payload_tol_frac: float = 0.35

    # ---- the MVC-0 admission gate (clu-controller-spec §C3/§C5/§4) ----
    # d_safe = d_safe_mult * s. At 4.4 a Gaussian atom write contributes
    # exp(-4.4^2/2) = 6.3e-5 of its gradient scale at a neighbouring minimum.
    d_safe_mult: float = 4.4
    # C3 budget on the predicted fixed-point drift ||H^-1 grad dV(q*)|| of a
    # stored item. 0.1 is ~1/14 of the w20 ring's site spacing (1.414), i.e. well
    # inside the deadband where address error is measured to be free.
    delta_budget: float = 0.1
    c3_chunk_steps: int = 25  # write granularity at which C3 is re-checked
    n_relocation_candidates: int = 400
    proposal_radius: float = 2.0  # disk the controller proposes sites in

    # ---- item 1: the gate on the EXACT w20 failing setup ----
    interference_K: int = 4
    interference_write_steps: int = 300
    gate_rungs: List[str] = field(
        default_factory=lambda: ["designed", "sites_learned_payload", "free_mlp"]
    )
    gate_arms: List[str] = field(
        default_factory=lambda: ["ungated", "gated_spacing", "gated_c3", "anchored"]
    )
    gate_proposals: List[str] = field(default_factory=lambda: ["ring", "disk"])

    # ---- item 2: the sequential-write curve ----
    n_sequential_items: int = 16
    sequential_write_steps: int = 600
    # ⚠ free_mlp, NOT the w20 rung: sequential writes place items at arbitrary
    # gate-chosen locations in a disk, and `sites_learned_payload` carries a
    # DESIGNED K-well ring at radius f that would fight every off-ring site.
    # The w20 rung is used where the w20 geometry is used -- item 1.
    sequential_rung: str = "free_mlp"
    sequential_arms: List[str] = field(
        default_factory=lambda: [
            "designed_gated",
            "designed_ungated",
            "learned_gated",
            "learned_ungated",
            # "anchored" is NOT a gate: it is a structured write operator
            # (C3 option (b), rehearsal from the codebook), carried as the arm
            # that shows what actually rescues a global-support write.
            "learned_anchored",
        ]
    )
    # designed store (AtomStorePotential; theorist S3 values)
    atom_width: float = 0.35
    atom_alpha: float = 0.02
    atom_amp: float = 1.0

    # ---- item 3: the cross-primitive comparison (primitive-harness slot) ----
    kv_primitives: List[str] = field(
        default_factory=lambda: ["mlp", "gru", "attention", "clu"]
    )
    kv_n_items: int = 16
    kv_select_items: int = 8  # shorter run used for the (equal) LR selection
    # >= kv_extended_items + 1: values are sampled WITHOUT replacement so that
    # "retention" is per-item unambiguous.
    kv_vocab: int = 128
    kv_key_len: int = 4
    kv_max_write_steps: int = 200
    kv_check_every: int = 1  # 1 => full resolution on compute-to-criterion
    # A second, LONGER sweep at the selected LR only: K=16 is matched to the CLU
    # arm, but if a primitive does not forget at all there, the curve is
    # uninformative and the interesting quantity is WHERE it breaks.
    kv_extended_items: int = 64
    # ...and give the extended sweep its own symmetric rescue: the LR selected at
    # K=16 need not be the best one at K=64, and the extended table is the
    # deliverable, so it must not inherit an unrescued LR.
    kv_extended_rescue: bool = True

    # ---- item 4: retrieval cost scaling in K ----
    cost_K_grid: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])


@dataclass
class ExperimentHopfieldCapacityConfig:
    """Configuration for the Hopfield-capacity PERFORMANCE benchmark (w22).

    The ONE external benchmark where a *designed* CLU is admissible (scout
    #1-ranked target): the modern-Hopfield / dense-associative-memory retrieval
    protocol, matched VERBATIM to ``MAGICS-LAB/UHop @ cdac754``
    (``memory_retrieval.py``) and ``ml-jku/hopfield-layers @ f56f929``. Nothing
    is learned on either side (the Hopfield line writes patterns in closed form;
    CLU designs a landscape), so the w20 "learning destroys everything" blocker
    does not bind. See chlu/experiments/exp_hopfield_capacity.py.

    ⚠ The repo's success metric is **mean sqdiff** (Σ(clamp(x,0,1)−clamp(x̂,0,1))²
    over pixels), NOT cosine>0.9 — the scout's cosine number was single-sourced.
    We match sqdiff and ALSO report cosine + identity-retrieval accuracy.
    """

    datasets: List[str] = field(default_factory=lambda: ["mnist"])
    load_grid: List[int] = field(default_factory=lambda: [16, 32, 64, 128, 256, 512])
    noise_levels: List[float] = field(
        default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    noise_fixed_load: int = 128  # load at which the noise sweep is run
    n_data_pool: int = 2000  # how many images to draw the store from
    mask_p: float = 0.5  # dropout fraction (the "50% masked" query) — repo verbatim

    # Hopfield arms
    hopfield_beta: float = 1.0  # repo default (β=1, 1 step)
    hopfield_steps: int = 1  # repo default
    hopfield_beta_tuned: float = 0.0  # 0 ⇒ auto β·⟨x,x⟩≈200 (sharp) rule
    hopfield_steps_tuned: int = 2
    activations: List[str] = field(default_factory=lambda: ["softmax", "sparsemax"])

    # CLU designed register (GaussianMemoryPotential + damped Verlet).
    # well width s = clu_s_frac * median NN pattern distance. MUST be < ~0.4:
    # Gaussian wells wider than the inter-pattern spacing merge into one basin
    # and every query settles to the centroid (measured: acc 1.00 at 0.3 vs 0.17
    # = chance at 0.5 on 6 well-separated 32-d patterns). This is the localized
    # dense-associative-memory regime; it is NOT tuned per load.
    clu_s_frac: float = 0.3
    clu_b: float = 1.0
    clu_alpha: float = 1e-3
    clu_gamma: float = 0.1  # settling friction (address phase)
    clu_steps: int = 200
    clu_dt: float = 0.0  # 0 ⇒ auto-set for stability from s and b
    clu_tail_frac: float = 0.1  # read = mean of last tail_frac of the rollout
    clu_kinetic_mode: str = "newtonian_identity"

    # Retry differentiator (codebook-gated boosted second pass)
    retry_enabled: bool = True
    retry_boost: float = 1.5  # KE boost applied to the query on the retry launch
    retry_conf_frac: float = 0.5  # bottom-fraction of first-pass confidences retried

    success_cosine: float = 0.9  # scout's secondary criterion, reported alongside
    seed: int = 0
    rollout_chunk: int = 256


@dataclass
class ExperimentPhiReadInConfig:
    """Configuration for the learned read-in ``φ`` around a DESIGNED store (w23).

    The phase-doctrine flagship: *learn around a designed core*. A learned read-in
    ``φ: raw x → feature`` lifts the query into a representation space; the store is
    a **designed key–value register** (address = ``φ(x)`` written as a Gaussian
    well; payload = the raw ``x``), and read-out ``ψ`` = settle → return the
    payload. The w22 Hopfield/U-Hop protocol is re-fought in ``φ``-space against
    **kNN-in-φ** (the trivial baseline, now a fair fight — cf. the pixel-space NN
    floor that beat everyone in w22), **closed-form Hopfield-in-φ**, and the w22
    **raw-space CLU** line (continuity control). See chlu/experiments/
    exp_phi_read_in.py.

    ⚠ **Laundering control (Item 3, mandatory):** same ``φ``, trivial store swap.
    If kNN-in-φ matches CLU-in-φ everywhere, the win is ``φ``'s, not the store's
    (the C17-3 lesson). A CLU margin that exists ONLY with the designed store is
    the result the program needs.

    Two ``φ`` arms, both trained OFF the CLU side (w20's law):
      - ``pca``: frozen PCA-k (unsupervised, linear, cheap).
      - ``ae``:  a small autoencoder trained on a DISJOINT data-distribution pool
        with a reconstruction loss only — never sees the store, wells, or a
        retrieval loss.
    Both are fit on ``n_fit_pool`` images drawn disjoint from the store pool.

    Metric = mean ``sqdiff`` in **pixel space** on the returned payload (identical
    to w22 for comparability); identity-retrieval accuracy reported alongside.
    """

    datasets: List[str] = field(default_factory=lambda: ["mnist"])
    phi_arms: List[str] = field(default_factory=lambda: ["pca", "ae"])
    load_grid: List[int] = field(default_factory=lambda: [16, 32, 64, 128, 256])
    noise_levels: List[float] = field(
        default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    noise_fixed_load: int = 128
    n_data_pool: int = 1500  # store pool (patterns drawn from here)
    n_fit_pool: int = 3000  # DISJOINT pool used to fit/train φ
    mask_p: float = 0.5  # dropout fraction (the 50%-masked query) — repo verbatim

    # φ read-in
    phi_dim: int = 32  # feature dimension k (the store's address space)
    # φ-B autoencoder (trained on the disjoint pool, reconstruction only)
    ae_hidden: int = 256
    ae_epochs: int = 400  # full-batch Adam steps
    ae_lr: float = 1e-3
    ae_batch: int = 512

    # closed-form Hopfield-in-φ arm
    hopfield_beta: float = 1.0
    hopfield_steps: int = 1
    activations: List[str] = field(default_factory=lambda: ["softmax", "sparsemax"])

    # CLU designed register in φ-space (GaussianMemoryPotential + damped Verlet)
    clu_s_frac: float = 0.3  # s = clu_s_frac * median-NN(φ) distance (fixed rule)
    clu_b: float = 1.0
    clu_alpha: float = 1e-3
    clu_gamma: float = 0.1
    clu_steps: int = 200
    clu_dt: float = 0.0  # 0 ⇒ auto-set from s and b
    clu_tail_frac: float = 0.1
    clu_kinetic_mode: str = "newtonian_identity"

    # Item 4 — retry confidence probe (distance-to-nearest-well at settle)
    probe_retry_confidence: bool = True

    success_cosine: float = 0.9
    seed: int = 0
    rollout_chunk: int = 256


@dataclass
class ExperimentRetryComputeConfig:
    """Configuration for the RETRY-COMPUTE study (w23) — the accuracy-vs-compute
    curve for CLU retrieval, with five pre-registered controls.

    Promotes w22's single retry demo point (+46.9pp at ×1.5 compute) into a
    defensible curve. On the w22 retrieval protocol (Gaussian noise queries,
    matched to ``MAGICS-LAB/UHop`` ``memory_retrieval_noise.py``), sweeps a retry
    ladder k ∈ {0,1,2,4,8} at ≥2 loads (M) and ≥2 noise levels (σ), reporting
    identity accuracy vs total relaxation-step count for six lines:
    CLU-gated retry, ungated-retry-all, ensemble-of-k-reads, random-kick retry,
    feedforward-NN matched-compute, Hopfield-k-steps.
    See chlu/experiments/exp_retry_compute.py.
    """

    datasets: List[str] = field(default_factory=lambda: ["mnist"])
    # Loads/σ chosen for HEADROOM: first-pass CLU accuracy in [0.2, 0.9] (measured
    # M=128/σ=0.3→0.69, M=256/σ=0.3→0.22, M=*/σ=0.2→0.83-0.92). Beyond the σ≈0.4
    # basin-capture cliff every arm collapses to chance and no retry can recover a
    # query that has left every well (the Δ_req limit — hopfield-capacity §2.2).
    load_grid: List[int] = field(default_factory=lambda: [128, 256])  # ≥2 loads
    noise_levels: List[float] = field(default_factory=lambda: [0.2, 0.3])  # ≥2 σ
    # Query protocol: "mask" = torch.dropout(p) (the w22 capacity protocol where the
    # retry boost recovers misses); "noise" = clamp(|x+N(0,σ)|,0,1) (Gaussian). The
    # boost is MASK-specific (measured); both are run so the finding travels.
    query_types: List[str] = field(default_factory=lambda: ["mask", "noise"])
    mask_fracs: List[float] = field(default_factory=lambda: [0.5, 0.7])  # ≥2 mask lvls
    n_data_pool: int = 2000
    retry_ladder: List[int] = field(default_factory=lambda: [0, 1, 2, 4, 8])
    retry_step_frac: float = 0.1  # fraction of lowest-conf reads retried per round
    # cosine gate thresholds swept for the gated line (Item 1: "threshold swept").
    # cos|correct≈0.998, cos|wrong≈0.95, so 0.99 cleanly gates the wrong tail;
    # 0.95 under-retries; 1.0 (no gate) over-retries and corrupts correct reads.
    conf_thresholds: List[float] = field(
        default_factory=lambda: [0.95, 0.97, 0.99, 1.0]
    )
    main_threshold: float = 0.99  # threshold used for the main-figure gated line
    retry_boost: float = 1.5  # KE boost injected toward the query on a retry launch

    # CLU designed register (mirrors ExperimentHopfieldCapacityConfig defaults)
    clu_s_frac: float = 0.3
    clu_b: float = 1.0
    clu_alpha: float = 1e-3
    clu_gamma: float = 0.1
    clu_steps: int = 150  # = 1 compute unit (one first-pass settle)
    clu_dt: float = 0.0  # 0 ⇒ auto-set for stability from s and b
    clu_tail_frac: float = 0.1
    clu_kinetic_mode: str = "newtonian_identity"

    # feedforward-NN matched-compute control (TTA augmentation + majority vote)
    ff_aug_sigma: float = 0.1
    # Hopfield-k-steps control (closed-form line iterated)
    hopfield_beta: float = 1.0
    hopfield_beta_tuned: float = 0.0  # 0 ⇒ auto β·⟨x,x⟩≈200 (floored at repo β)

    seed: int = 0
    rollout_chunk: int = 256


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
    experiment_v1_wormhole: ExperimentV1WormholeConfig = field(
        default_factory=ExperimentV1WormholeConfig
    )
    experiment_lattice: ExperimentLatticeConfig = field(
        default_factory=ExperimentLatticeConfig
    )
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
        experiment_v1_wormhole=ExperimentV1WormholeConfig(
            **filter_valid_fields(
                ExperimentV1WormholeConfig, data.get("experiment_v1_wormhole", {})
            )
        ),
        experiment_lattice=ExperimentLatticeConfig(
            **filter_valid_fields(
                ExperimentLatticeConfig, data.get("experiment_lattice", {})
            )
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
                ExperimentMinusPhysicsConfig,
                data.get("experiment_minus_physics", {}),
            )
        ),
        experiment_kt=ExperimentKTConfig(
            **filter_valid_fields(ExperimentKTConfig, data.get("experiment_kt", {}))
        ),
        experiment_retrieval=ExperimentRetrievalConfig(
            **filter_valid_fields(
                ExperimentRetrievalConfig, data.get("experiment_retrieval", {})
            )
        ),
        experiment_dim_scaling=ExperimentDimScalingConfig(
            **filter_valid_fields(
                ExperimentDimScalingConfig, data.get("experiment_dim_scaling", {})
            )
        ),
        experiment_learned_memory=ExperimentLearnedMemoryConfig(
            **filter_valid_fields(
                ExperimentLearnedMemoryConfig,
                data.get("experiment_learned_memory", {}),
            )
        ),
        experiment_potential_class=ExperimentPotentialClassConfig(
            **filter_valid_fields(
                ExperimentPotentialClassConfig,
                data.get("experiment_potential_class", {}),
            )
        ),
        experiment_designed_mechanism=ExperimentDesignedMechanismConfig(
            **filter_valid_fields(
                ExperimentDesignedMechanismConfig,
                data.get("experiment_designed_mechanism", {}),
            )
        ),
        experiment_primitive_harness=ExperimentPrimitiveHarnessConfig(
            **filter_valid_fields(
                ExperimentPrimitiveHarnessConfig,
                data.get("experiment_primitive_harness", {}),
            )
        ),
        experiment_sequential_write=ExperimentSequentialWriteConfig(
            **filter_valid_fields(
                ExperimentSequentialWriteConfig,
                data.get("experiment_sequential_write", {}),
            )
        ),
        experiment_hopfield_capacity=ExperimentHopfieldCapacityConfig(
            **filter_valid_fields(
                ExperimentHopfieldCapacityConfig,
                data.get("experiment_hopfield_capacity", {}),
            )
        ),
        experiment_phi_read_in=ExperimentPhiReadInConfig(
            **filter_valid_fields(
                ExperimentPhiReadInConfig,
                data.get("experiment_phi_read_in", {}),
            )
        ),
        experiment_retry_compute=ExperimentRetryComputeConfig(
            **filter_valid_fields(
                ExperimentRetryComputeConfig,
                data.get("experiment_retry_compute", {}),
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
        "experiment_v1_wormhole": asdict(config.experiment_v1_wormhole),
        "experiment_lattice": asdict(config.experiment_lattice),
        "experiment_s1": asdict(config.experiment_s1),
        "experiment_minus_physics": asdict(config.experiment_minus_physics),
        "experiment_paid_access": asdict(config.experiment_paid_access),
        "experiment_kt": asdict(config.experiment_kt),
        "experiment_retrieval": asdict(config.experiment_retrieval),
        "experiment_dim_scaling": asdict(config.experiment_dim_scaling),
        "experiment_learned_memory": asdict(config.experiment_learned_memory),
        "experiment_potential_class": asdict(config.experiment_potential_class),
        "experiment_designed_mechanism": asdict(config.experiment_designed_mechanism),
        "experiment_primitive_harness": asdict(config.experiment_primitive_harness),
        "experiment_sequential_write": asdict(config.experiment_sequential_write),
        "experiment_hopfield_capacity": asdict(config.experiment_hopfield_capacity),
        "experiment_phi_read_in": asdict(config.experiment_phi_read_in),
        "experiment_retry_compute": asdict(config.experiment_retry_compute),
        "data": asdict(config.data),
        "project": asdict(config.project),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
