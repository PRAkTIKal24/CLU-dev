"""Configuration dataclasses for the evaluation harness.

Kept self-contained (not in ``chlu/config.py``) deliberately: the harness is
model-agnostic infrastructure, and ``chlu/config.py`` is under concurrent
modification by other work streams. All knobs are explicit here — no magic
numbers in run bodies.
"""

from dataclasses import asdict, dataclass, field
import json


@dataclass(frozen=True)
class WindowConfig:
    """Explicit sliding-window configuration (binding rule: windowing explicit).

    Attributes:
        size: Window length in samples.
        stride: Stride between consecutive *test* windows. 1 gives per-point
            scores after overlap-averaging.
        train_stride: Stride for *training* windows (>=1). Larger values
            subsample the training set — record it in reports.
    """

    size: int = 100
    stride: int = 1
    train_stride: int = 1

    def __post_init__(self) -> None:
        if self.size < 2:
            raise ValueError(f"window size must be >= 2, got {self.size}")
        if self.stride < 1 or self.train_stride < 1:
            raise ValueError("strides must be >= 1")


@dataclass(frozen=True)
class EvalConfig:
    """Configuration for one harness run (one dataset).

    Attributes:
        window: Sliding-window settings for the statistical baselines.
        metrics_sliding_window: ``slidingWindow`` handed to TSB-AD's VUS
            implementation (the buffer-region half-width). TSB-AD's own default
            is 100; set per dataset and report it.
        metrics_mode: "full" = TSB-AD ``get_metrics`` set (VUS/AUC + the
            threshold-optimised F1 family); "fast" = threshold-independent
            metrics only (VUS-PR/VUS-ROC/AUC-PR/AUC-ROC) — for large sweeps.
        episode_reduce: How per-point scores are pooled into one score for
            episode-labelled datasets (e.g. voraus-AD): "mean" or "max".
        seed: RNG seed for the seeded baselines and any subsampling.
        max_train_windows: Memory guard — training windows are subsampled
            (seeded, uniform) beyond this count. ``None`` disables.
        pca_n_components: PCA-recon components (int) or retained-variance
            fraction (float in (0, 1)).
        iforest_n_estimators: IsolationForest size.
        lof_n_neighbors: LOF neighbourhood size.
        knn_n_neighbors: KNN neighbourhood size (score = mean distance to the
            k nearest training windows).
    """

    window: WindowConfig = field(default_factory=WindowConfig)
    metrics_sliding_window: int = 100
    metrics_mode: str = "full"
    episode_reduce: str = "mean"
    seed: int = 42
    max_train_windows: int | None = 100_000
    pca_n_components: float | int = 0.9
    iforest_n_estimators: int = 100
    lof_n_neighbors: int = 20
    knn_n_neighbors: int = 10

    def __post_init__(self) -> None:
        if self.metrics_mode not in ("full", "fast"):
            raise ValueError(f"metrics_mode must be full|fast, got {self.metrics_mode}")
        if self.episode_reduce not in ("mean", "max"):
            raise ValueError(
                f"episode_reduce must be mean|max, got {self.episode_reduce}"
            )
        if self.metrics_sliding_window < 1:
            raise ValueError("metrics_sliding_window must be >= 1")

    def to_json(self) -> str:
        """Serialize (for embedding into results files — provenance)."""
        return json.dumps(asdict(self), sort_keys=True)


#: Anomaly-score arms the CLU scorer supports.
#:   ``energy``   — H(q, p) of the (finite-difference-momentum) window states
#:                  after the learned potential is fitted (EBM reading:
#:                  anomalies are high-energy states off the data basin).
#:   ``residual`` — relaxation residual R0: run a short damped rollout from
#:                  each window state and measure how badly it fails to settle
#:                  into a basin (mean squared force + energy the window fails
#:                  to shed). The dissipation-proof-memory reading of §1.4.
#:   ``predict``  — multi-step prediction MSE of the CLU rollout over the
#:                  window (the conventional TSAD framing; the fairest
#:                  head-to-head vs PCA-recon).
#:   ``hybrid``   — a documented, UNTUNED combination (energy-gated prediction
#:                  error). Hook only — do not tune in the first pass; it is
#:                  the final iteration, tried only if it beats both singles.
#: FUTURE: settling-time score (handover 2026-07-19) — number of rollout steps
#: to fall into a latch; explored only after the first CSF results, may be
#: dropped entirely. A third first-class arm would slot in here.
CLU_SCORE_MODES = ("energy", "residual", "predict", "hybrid")

#: The two mandatory first-pass arms (both first-class, compared from the data;
#: Head 2026-07-19). ``residual`` is the variant of the energy arm.
CLU_DEFAULT_SCORE_MODES = ("energy", "residual", "predict")


@dataclass(frozen=True)
class CLULatticeConfig:
    """Optional CLU-lattice hook (G7b torus-coset prerequisite).

    When a ``CLUScorerConfig`` carries a ``lattice``, the scorer fits a
    :class:`chlu.core.lattice.CLULattice` instead of a single ``CHLU``. This
    task only wires the *hook*: units are laid out to tile the C data channels
    (``unit_dim`` channels per unit), and the coupling/topology knobs are
    forwarded to :func:`chlu.core.lattice.build_lattice`. The literal
    joint-angle -> so2_invariant-coset mapping (each voraus joint angle to one
    ``so2_invariant`` unit's U(1) coset — the falsifiable "n independent
    dissipation-proof registers" prediction) is the next task
    (``g7b-torus-voraus``); here we only guarantee the config is expressible.

    Attributes:
        unit_dim: channels handled per unit (2 = one SO(2) coset per unit —
            the literal torus mapping's atom). C must be divisible by this.
        topology: ``"chain"`` (default) or ``"torus"`` (needs n_units = L^2).
        coupling_type: forwarded to ``build_lattice`` ("auto" picks
            channel_spring for so2_invariant units — the U(1)-preserving choice).
        kappa_c: coupling strength.
        coupling_dim: coupling channel width (2 for the SO(2) coset).
        potential_type: per-unit potential ("mlp" or "so2_invariant").
        tie_channel_mass: kinetic isotropy on each unit's SO(2) channel.
    """

    unit_dim: int = 2
    topology: str = "chain"
    coupling_type: str = "auto"
    kappa_c: float = 0.05
    coupling_dim: int = 2
    potential_type: str = "so2_invariant"
    tie_channel_mass: bool = True

    def __post_init__(self) -> None:
        if self.unit_dim < 1:
            raise ValueError(f"unit_dim must be >= 1, got {self.unit_dim}")
        if self.topology not in ("chain", "torus"):
            raise ValueError(f"topology must be chain|torus, got {self.topology}")


@dataclass(frozen=True)
class CLUScorerConfig:
    """Configuration for the CLU anomaly scorer (``CHLUScorer``).

    Kept here (not in ``chlu/config.py``) on purpose — the eval scorer is
    harness infrastructure and ``chlu/config.py`` is under concurrent edit.
    All knobs explicit; defaults chosen so a laptop smoke fits quickly and
    produces finite scores (score *quality* is a downstream science question).

    Attributes:
        kinetic_mode: CHLU kinetic governor ("newtonian_identity",
            "newtonian_learned", "relativistic").
        potential_type: single-unit potential ("mlp", "deep_mlp",
            "so2_invariant"). Ignored when ``lattice`` is set.
        hidden: potential-net hidden width.
        rest_mass, c: relativistic-kinetics knobs (inert otherwise).
        tie_channel_mass: tie the SO(2) channel's inertial masses.
        dt: Verlet timestep for rollouts.
        gamma: dissipation used in the relaxation rollout (residual arm) and
            the predictive rollout.
        epochs: training epochs (Hamiltonian contrastive divergence).
        lr: Adam learning rate.
        batch_size: windows per gradient step.
        max_fit_windows: subsample cap on training windows fed to the CLU fit
            (memory/time guard; seeded uniform).
        predict_horizon: rollout length (steps) for the predict arm and its
            training loss (capped at window size - 1).
        relax_steps: damped-rollout length for the residual arm.
        residual_anchors: number of evenly-spaced window timesteps relaxed for
            the residual arm (caps residual cost per window).
        predict_weight: weight of the wake prediction-MSE loss term.
        energy_weight: weight of the contrastive energy term (push H(data)
            down, H(negatives) up).
        neg_noise_scale: Gaussian perturbation making the EBM negatives
            (denoising-EBM basin around the data manifold).
        energy_reg: energy-magnitude regularizer (keeps H from exploding;
            mirrors train_generative's 0.005 term).
        momentum_init: how p0 is seeded from a window ("finite_diff" =
            (q1-q0)/dt, "zero").
        seed: RNG seed for CLU init + training + subsampling.
        lattice: optional :class:`CLULatticeConfig` (G7b torus hook); None
            (default) fits a single ``CHLU``.
    """

    kinetic_mode: str = "newtonian_learned"
    potential_type: str = "mlp"
    hidden: int = 64
    rest_mass: float = 1.0
    c: float = 1.0
    tie_channel_mass: bool = False
    dt: float = 0.05
    gamma: float = 0.1
    epochs: int = 150
    lr: float = 1e-3
    batch_size: int = 64
    max_fit_windows: int = 4000
    predict_horizon: int = 16
    relax_steps: int = 32
    residual_anchors: int = 8
    predict_weight: float = 1.0
    energy_weight: float = 1.0
    neg_noise_scale: float = 0.5
    energy_reg: float = 0.005
    momentum_init: str = "finite_diff"
    seed: int = 42
    lattice: CLULatticeConfig | None = None

    def __post_init__(self) -> None:
        valid_kin = ("newtonian_identity", "newtonian_learned", "relativistic")
        if self.kinetic_mode not in valid_kin:
            raise ValueError(f"kinetic_mode must be one of {valid_kin}")
        if self.momentum_init not in ("finite_diff", "zero"):
            raise ValueError("momentum_init must be finite_diff|zero")
        if self.predict_horizon < 1 or self.relax_steps < 1:
            raise ValueError("predict_horizon and relax_steps must be >= 1")

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)
