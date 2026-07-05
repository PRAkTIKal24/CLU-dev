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
