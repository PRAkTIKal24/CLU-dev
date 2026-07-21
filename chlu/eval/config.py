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

    Two layouts (``layout``):

    - ``"tile"`` (default, the w15 hook): tile the C data channels into units of
      ``unit_dim`` channels each; requires an exact tiling (C % unit_dim == 0).
    - ``"literal"`` (G7b flagship): the LITERAL joint-angle -> so2 coset map.
      The first ``2 * n_so2_units`` channels are ``(cos θ_j, sin θ_j)`` pairs,
      each fed to one ``so2_invariant`` unit whose ``T^1`` coset *is* that
      joint's ``U(1)`` (produce them with ``VorausTorusAD``). The remaining
      ``C - 2 * n_so2_units`` channels are auxiliary (velocities, torques,
      temperatures) and are governed by plain-``mlp`` units tiled at
      ``aux_unit_dim`` (the LAST aux unit absorbs the non-divisible remainder —
      voraus's fixed column set will not be a clean multiple, and we do NOT
      pad). Only the so2 (angle) units sit on the coupling topology; aux units
      are isolated ("angles-on-the-torus, everything else auxiliary").

    Attributes:
        layout: ``"tile"`` or ``"literal"`` (see above).
        n_so2_units: (literal only) number of leading (cos, sin) angle pairs
            = number of ``so2_invariant`` coset units (6 for voraus's 6 axes).
        aux_unit_dim: (literal only) channels per auxiliary ``mlp`` unit; the
            final aux unit takes the remainder (no padding).
        unit_dim: (tile only) channels handled per unit (2 = one SO(2) coset).
            C must be divisible by this in tile layout.
        topology: ``"chain"`` (default), ``"ring"`` (chain closed into a 1-D
            torus — the honest topology for the 6-axis SERIAL robot's
            kinematic chain), or ``"torus"`` (2-D, needs n_units = L^2).
        shuffle_angles: (literal only) TOPOLOGY-MATCH CONTROL — randomly
            permute which coset each edge connects, destroying the
            kinematic-chain adjacency. If the topology match matters, this
            must do WORSE (pre-registered falsifier). ``shuffle_seed`` seeds it.
        coupling_type: forwarded to the coupling builder ("auto"/"channel_spring"
            keep U(1) intact; the random-W ``spring`` breaks it — CM-9).
        kappa_c: coupling strength.
        coupling_dim: coupling channel width (2 for the SO(2) coset).
        potential_type: per-unit potential in tile layout ("mlp" or
            "so2_invariant"). In literal layout the angle units are always
            ``so2_invariant`` and the aux units always ``mlp``.
        tie_channel_mass: kinetic isotropy on each unit's SO(2) channel.
    """

    unit_dim: int = 2
    topology: str = "chain"
    coupling_type: str = "auto"
    kappa_c: float = 0.05
    coupling_dim: int = 2
    potential_type: str = "so2_invariant"
    tie_channel_mass: bool = True
    layout: str = "tile"
    n_so2_units: int = 0
    aux_unit_dim: int = 4
    shuffle_angles: bool = False
    shuffle_seed: int = 0

    def __post_init__(self) -> None:
        if self.unit_dim < 1:
            raise ValueError(f"unit_dim must be >= 1, got {self.unit_dim}")
        if self.topology not in ("chain", "ring", "torus"):
            raise ValueError(f"topology must be chain|ring|torus, got {self.topology}")
        if self.layout not in ("tile", "literal"):
            raise ValueError(f"layout must be tile|literal, got {self.layout}")
        if self.layout == "literal":
            if self.n_so2_units < 1:
                raise ValueError("literal layout needs n_so2_units >= 1")
            if self.aux_unit_dim < 1:
                raise ValueError("aux_unit_dim must be >= 1")


#: Feature groups the CAFE encoder can emit (see :class:`CLUCafeEncodeConfig`).
#: Each group is a *physics* read of the trained CLU, not a learned projection:
#:   ``energy``       — H(q, p) level/spread/TREND over the window. The trend is
#:                      the basin-exit signal: a system drifting off its basin
#:                      accumulates energy monotonically.
#:   ``potential``    — V(q) level/trend (the landscape height alone).
#:   ``kinetic``      — T(p) = H - V level (how fast the state is moving).
#:   ``gradv``        — ||grad V(q)||^2 level/last/trend. The DISTANCE-TO-
#:                      INSTABILITY read: at a basin bottom ||grad V|| ~ 0, and
#:                      it grows as the state is pushed up the valley wall.
#:   ``relax``        — damped rollout from the window's FINAL state to a
#:                      settled point q*: relaxation residual ||grad V(q*)||^2,
#:                      the settled height V(q*), and the drift ||q* - q_end||^2
#:                      (how far the state had to fall to reach its basin).
#:                      This is the basin-MEMBERSHIP / valley-aware read.
#:   ``predict``      — multi-step CLU rollout MSE over the window.
#:   ``basin_coords`` — the settled point q* itself (C dims): WHICH basin the
#:                      window belongs to, not just how far from it.
CAFE_FEATURE_GROUPS = (
    "energy",
    "potential",
    "kinetic",
    "gradv",
    "relax",
    "predict",
    "basin_coords",
)

#: Valley-aware anomaly arms (item 3 of ``clu-cafe-integration``): score the
#: SETTLED state, not the transient. ``valley`` = relaxation residual +
#: V(q*); ``valley_predict`` additionally adds the (z-scored) rollout error.
CAFE_ANOMALY_MODES = ("valley", "valley_predict", "energy", "predict")


@dataclass(frozen=True)
class CLUCafeEncodeConfig:
    """Configuration for the CAFE ``encode()`` feature map (CLU -> (N, D)).

    The CAFE harness contract is ``encode(X: (N, T, C)) -> (N, D)`` frozen
    embeddings, which a downstream probe then fits (``BaseModel``:
    LogisticRegression / kNN / CoxPH). This config selects WHICH physics reads
    of the trained CLU become embedding coordinates.

    ⚠ Why the feature choice is load-bearing for Event Prediction: CAFE's
    default event probe is a **linear** CoxPH on the embedding, and a
    proportional-hazards risk score ``beta . z`` induces the SAME sample
    ranking at every horizon. So h-AUROC is determined entirely by one linear
    functional of these features — the embedding must therefore expose
    coordinates that are individually monotone in degradation, not merely
    informative in a nonlinear sense.

    Attributes:
        feature_groups: which of :data:`CAFE_FEATURE_GROUPS` to emit, in order.
        standardize: z-score the raw windows with statistics fitted on the
            first (training) ``encode`` call. CAFE's C-MAPSS loader already
            normalizes per channel, so this is near a no-op there; it protects
            datasets whose loaders do not normalize.
        batch_size: windows per vmapped feature-extraction chunk (memory guard).
        anomaly_mode: which arm :meth:`CLUCafeModel.anomaly_score` uses.
        max_probe_train: cap on training rows handed to the downstream probe
            (CoxPH on ~18k x D is the dominant cost of a C-MAPSS run);
            ``None`` disables. Subsampling is seeded and uniform.
        relax_gamma, relax_steps: OPTIONAL overrides of the ``CLUScorerConfig``
            dissipation/rollout length, used only by the CAFE feature map.
            ``None`` inherits. These exist as separate knobs because the
            anomaly scorer's defaults (gamma=0.1, relax_steps=32) are tuned for
            a different job and are shared with the voraus path — changing them
            there would move published anomaly numbers.

            ⚠ CORRECTED 2026-07-20 (`cmapss-fd002-004-fetch`). An earlier note
            here claimed "the relaxation BUDGET ``gamma * steps * dt`` is the
            knob that matters, not either factor alone". **That is false, and a
            2-D (gamma, steps) grid falsifies it directly:**

              * ISO-GAMMA, 400x budget range (gamma=0.5, steps 6 -> 2560,
                budget 0.15 -> 64): h-AUROC **0.7230 at every point**, identical
                to 4 decimals on FD001 (0.6537/0.6537/0.6536/0.6534 on FD002).
              * ISO-BUDGET (budget=1.6, gamma 0.05/0.1/0.2/0.5): h-AUROC
                **0.6109 / 0.6744 / 0.7132 / 0.7230** on FD001 — a 0.11 spread.

            ⇒ **gamma alone controls the result; steps is very nearly inert.**
            Performance is monotone increasing in gamma over the tested range,
            because larger gamma freezes the state faster, so q* stays closer to
            the window's last observation (corr(q*, q_last) rises 0.56 -> 0.92
            as gamma goes 0.05 -> 0.5, and h-AUROC rises with it). The best
            setting is the one that does the LEAST dynamics; plain ``q_last``
            with no CLU at all scores 0.7203 vs CLU's best 0.7230.

            ⚠ ``relax_gamma > 2`` DIVERGES. The dissipative step is
            ``p <- (1 - gamma) * p``, so |1 - gamma| > 1 amplifies momentum
            every step and the rollout overflows within ~50 steps. Because
            :meth:`CLUCafeMixin.encode` zero-fills non-finite rows, this
            presents as a cross-sample spread of EXACTLY 0.000 plus a singular
            probe — indistinguishable from a physical basin collapse, and
            previously misreported as one. Measured: gamma >= 5 gives 100%
            non-finite rows; gamma = 0.5 and 2.0 give 0%.
    """

    feature_groups: tuple = CAFE_FEATURE_GROUPS
    standardize: bool = True
    batch_size: int = 512
    anomaly_mode: str = "valley"
    max_probe_train: int | None = None
    relax_gamma: float | None = None
    relax_steps: int | None = None

    def relax_budget(self, cfg) -> float:
        """The dimensionless damping budget ``gamma * steps * dt_eff`` in use.

        ``relax_steps`` counts INTEGRATOR steps, so the budget is set by the
        integrator step ``dt_eff``, not by ``data_dt``. Note this rescaled with
        the w20 dt-units split: at the old conflated ``dt=0.05`` the default
        budget was 0.16, at ``dt_eff=1.0`` the same ``gamma/steps`` gives 3.2.
        """
        gamma = self.relax_gamma if self.relax_gamma is not None else cfg.gamma
        steps = self.relax_steps if self.relax_steps is not None else cfg.relax_steps
        return float(gamma) * int(steps) * float(cfg.dt_eff)

    def __post_init__(self) -> None:
        if self.relax_steps is not None and self.relax_steps < 1:
            raise ValueError("relax_steps override must be >= 1")
        if self.relax_gamma is not None and self.relax_gamma < 0:
            raise ValueError("relax_gamma override must be >= 0")
        for g in self.feature_groups:
            if g not in CAFE_FEATURE_GROUPS:
                raise ValueError(
                    f"unknown CAFE feature group {g!r}; valid: {CAFE_FEATURE_GROUPS}"
                )
        if not self.feature_groups:
            raise ValueError("feature_groups must be non-empty")
        if self.anomaly_mode not in CAFE_ANOMALY_MODES:
            raise ValueError(
                f"anomaly_mode must be one of {CAFE_ANOMALY_MODES}, "
                f"got {self.anomaly_mode}"
            )
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


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
        dt: Verlet timestep — the INTEGRATOR step, a numerical-accuracy knob.
            Constrained only by stability (``dt * omega < 2``); it does NOT
            carry the data's time units. See ``data_dt``.

            ⚠ **Choose this self-consistently, and do not trust an at-init
            check.** ``omega = sqrt(lambda_max(grad^2 V) / M_min)`` is a
            property of the TRAINED model, and the trained curvature depends on
            the ``dt`` it was trained at. Measured on FD001 (150 epochs, seed
            42), ``omega`` at init is 0.51 -- which makes ``dt=1.0`` look safe
            -- but after training:

                dt=1.0  -> omega= 4.13, dt*omega=4.13  UNSTABLE
                dt=0.5  -> omega= 5.54, dt*omega=2.77  UNSTABLE
                dt=0.25 -> omega= 7.96, dt*omega=1.99  marginal
                dt=0.125-> omega=13.46, dt*omega=1.68  (default, 1.19x margin)
                dt=0.05 -> omega=22.32, dt*omega=1.12  (1.79x margin, 20x cost)

            ``omega`` GROWS as ``dt`` shrinks: a finer integrator lets training
            build a sharper potential, which eats the margin it just bought.
            Nothing in the objective penalizes curvature, so the model
            self-organizes to the stability edge and NO ``dt`` buys a
            comfortable margin. Lowering ``dt`` still helps (energy drift over
            16 cycles falls 2.94 -> 0.081) but sub-linearly, at ``substeps``x
            the compute. The real fix is curvature control or a mass floor.
            ``mass_spread_lambda`` makes this WORSE from the mass side
            (lambda=50 drives ``M_min`` to 0.027 and ``dt*omega`` to 29.6), so
            retune ``dt`` whenever you change it.
        data_dt: the PHYSICAL sampling interval of the data, in the data's own
            time units — the Delta-t separating consecutive window frames. Used
            for (a) the finite-difference momentum ``p = (q1-q0)/data_dt`` and
            (b) fixing how much physical time one predicted sample spans.
            On cycle-indexed C-MAPSS one frame is one cycle, so **1.0**.

            ⚠ These two were a SINGLE field (``dt=0.05``) until w20, which
            conflated a physical unit with a numerical one. At 0.05 on
            cycle-indexed data the momenta were inflated 20x and K by 400x,
            which made the ``energy_reg`` term 99.2% of the loss and left the
            wake rollout 98.3% ballistic (``clu-latent-io-audit``, w19). Setting
            ``data_dt == dt`` reproduces the old (conflated) behaviour exactly.
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
        mass_lr_mult: run every ``log_mass`` leaf on its own Adam slot at
            ``lr * mass_lr_mult`` (mirrors ``training.mass_lr_mult``, which was
            wired ONLY into ``chlu/training/train.py`` and therefore never
            reached this — the CAFE/eval — training path). Default 1.0 is
            bit-compatible (plain ``optax.adam(lr)``, no ``multi_transform``).
            ⚠ CM-5/N8: 10x is the known-safe setting; 100x inverts the ordering.
        mass_spread_lambda: coefficient of the R-1 mass-spread term
            ``-lambda * Var(log_mass)`` added to the training loss, i.e. an
            explicit pressure toward a NON-DEGENERATE timescale hierarchy
            (the "hierarchy must be designed in" doctrine, CM-5). Default 0.0 =
            OFF = term never touched. Applied from epoch 0 (theorist T3).
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
    dt: float = 0.125
    data_dt: float = 1.0
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
    mass_lr_mult: float = 1.0
    mass_spread_lambda: float = 0.0
    seed: int = 42
    lattice: CLULatticeConfig | None = None

    def __post_init__(self) -> None:
        valid_kin = ("newtonian_identity", "newtonian_learned", "relativistic")
        if self.kinetic_mode not in valid_kin:
            raise ValueError(f"kinetic_mode must be one of {valid_kin}")
        if self.momentum_init not in ("finite_diff", "zero"):
            raise ValueError("momentum_init must be finite_diff|zero")
        if self.mass_lr_mult <= 0:
            raise ValueError("mass_lr_mult must be > 0")
        if self.mass_spread_lambda < 0:
            raise ValueError("mass_spread_lambda must be >= 0")
        if self.predict_horizon < 1 or self.relax_steps < 1:
            raise ValueError("predict_horizon and relax_steps must be >= 1")
        if self.dt <= 0 or self.data_dt <= 0:
            raise ValueError("dt and data_dt must be > 0")
        if self.dt > self.data_dt + 1e-12:
            raise ValueError(
                f"dt ({self.dt}) must not exceed data_dt ({self.data_dt}): the "
                "integrator cannot take a step longer than the interval it is "
                "asked to resolve"
            )

    @property
    def substeps(self) -> int:
        """Integrator steps per DATA sample: ``round(data_dt / dt)``, min 1.

        A rollout that is compared against data must advance exactly one
        ``data_dt`` per predicted sample; with a finer integrator step that
        takes ``substeps`` Verlet steps per sample. ``dt == data_dt`` gives 1
        and the plain single-step rollout (the pre-w20 code path).
        """
        return max(1, int(round(self.data_dt / self.dt)))

    @property
    def dt_eff(self) -> float:
        """The integrator step actually used: ``data_dt / substeps``.

        Snapped so that ``substeps`` steps land exactly on the data grid — an
        un-snapped ``dt`` would accumulate a physical-time offset over a rollout.
        """
        return self.data_dt / self.substeps

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)
