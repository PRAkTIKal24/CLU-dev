"""Mandatory statistical baselines + explicit windowing utilities.

Binding rule (Quo Vadis ICML'24; TSB-AD NeurIPS'24): every results table must
include simple statistical baselines — deep TSAD models are frequently matched
by them. The four here (PCA-reconstruction, IsolationForest, LOF, KNN) are
sklearn implementations wired into every ``evaluate_dataset`` run by default.

Protocol: semi-supervised (fit on normal/train windows only, score test
windows). All baselines consume flattened sliding windows of the multivariate
series and emit higher-is-more-anomalous window scores, which
``window_scores_to_point_scores`` maps back to per-timestep scores by
overlap-averaging.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from chlu.eval.config import EvalConfig


def sliding_windows(x: np.ndarray, size: int, stride: int = 1) -> np.ndarray:
    """Cut a (T, C) series into flattened windows.

    Args:
        x: (T,) or (T, C) array.
        size: Window length (must satisfy T >= size).
        stride: Step between window starts.

    Returns:
        (n_windows, size * C) float32 array, n_windows = (T - size)//stride + 1.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"expected (T,) or (T, C), got shape {x.shape}")
    if len(x) < size:
        raise ValueError(f"series length {len(x)} < window size {size}")
    windows = sliding_window_view(x, (size, x.shape[1]))[::stride, 0]
    return windows.reshape(windows.shape[0], -1)


def window_scores_to_point_scores(
    window_scores: np.ndarray, n_points: int, size: int, stride: int = 1
) -> np.ndarray:
    """Map per-window scores back to per-timestep scores by overlap-averaging.

    Every timestep receives the mean score of all windows covering it; tail
    timesteps not covered by any window (possible when stride > 1) inherit the
    last covered value. Explicit and deterministic by design.
    """
    window_scores = np.asarray(window_scores, dtype=np.float64).ravel()
    acc = np.zeros(n_points)
    cnt = np.zeros(n_points)
    for i, s in enumerate(window_scores):
        start = i * stride
        acc[start : start + size] += s
        cnt[start : start + size] += 1
    covered = cnt > 0
    if not covered.any():
        raise ValueError("no timestep covered by any window")
    point = np.empty(n_points)
    point[covered] = acc[covered] / cnt[covered]
    if not covered.all():  # forward/backward-fill the uncovered tail/head
        idx = np.where(covered)[0]
        point[: idx[0]] = point[idx[0]]
        point[idx[-1] :] = point[idx[-1]]
    return point


class BaselineScorer(ABC):
    """Semi-supervised window scorer: fit on normal windows, score test ones."""

    name: str = "baseline"

    @abstractmethod
    def fit(self, train_windows: np.ndarray) -> "BaselineScorer":
        """Fit on (n, d) training (normal) windows."""

    @abstractmethod
    def score(self, windows: np.ndarray) -> np.ndarray:
        """Return (n,) anomaly scores, higher = more anomalous."""


class PCAReconBaseline(BaselineScorer):
    """PCA reconstruction error (the classic subspace baseline)."""

    name = "pca_recon"

    def __init__(self, n_components: float | int = 0.9, seed: int = 42):
        self.n_components = n_components
        self.seed = seed
        self._pca = None

    def fit(self, train_windows: np.ndarray) -> "PCAReconBaseline":
        from sklearn.decomposition import PCA

        n_comp = self.n_components
        max_comp = min(train_windows.shape)
        if isinstance(n_comp, (int, np.integer)):
            n_comp = int(min(n_comp, max_comp))
        self._pca = PCA(
            n_components=n_comp, svd_solver="full", random_state=self.seed
        ).fit(train_windows)
        return self

    def score(self, windows: np.ndarray) -> np.ndarray:
        recon = self._pca.inverse_transform(self._pca.transform(windows))
        return np.mean((windows - recon) ** 2, axis=1)


class IForestBaseline(BaselineScorer):
    """IsolationForest (sklearn), seeded."""

    name = "iforest"

    def __init__(self, n_estimators: int = 100, seed: int = 42):
        self.n_estimators = n_estimators
        self.seed = seed
        self._model = None

    def fit(self, train_windows: np.ndarray) -> "IForestBaseline":
        from sklearn.ensemble import IsolationForest

        self._model = IsolationForest(
            n_estimators=self.n_estimators, random_state=self.seed
        ).fit(train_windows)
        return self

    def score(self, windows: np.ndarray) -> np.ndarray:
        return -self._model.score_samples(windows)


class LOFBaseline(BaselineScorer):
    """Local Outlier Factor in novelty mode (fit on normal data only)."""

    name = "lof"

    def __init__(self, n_neighbors: int = 20):
        self.n_neighbors = n_neighbors
        self._model = None

    def fit(self, train_windows: np.ndarray) -> "LOFBaseline":
        from sklearn.neighbors import LocalOutlierFactor

        k = int(min(self.n_neighbors, max(1, len(train_windows) - 1)))
        self._model = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(train_windows)
        return self

    def score(self, windows: np.ndarray) -> np.ndarray:
        return -self._model.score_samples(windows)


class KNNBaseline(BaselineScorer):
    """K-nearest-neighbour distance to the training set."""

    name = "knn"

    def __init__(self, n_neighbors: int = 10):
        self.n_neighbors = n_neighbors
        self._model = None
        self._k = None

    def fit(self, train_windows: np.ndarray) -> "KNNBaseline":
        from sklearn.neighbors import NearestNeighbors

        self._k = int(min(self.n_neighbors, len(train_windows)))
        self._model = NearestNeighbors(n_neighbors=self._k).fit(train_windows)
        return self

    def score(self, windows: np.ndarray) -> np.ndarray:
        dist, _ = self._model.kneighbors(windows, n_neighbors=self._k)
        return dist.mean(axis=1)


def make_default_baselines(config: EvalConfig) -> dict:
    """The four mandatory statistical baselines, from explicit config.

    Every evaluation run includes these by default (binding rule). Additional
    scorers may be *added* by callers, but these four are the floor.
    """
    return {
        "pca_recon": PCAReconBaseline(
            n_components=config.pca_n_components, seed=config.seed
        ),
        "iforest": IForestBaseline(
            n_estimators=config.iforest_n_estimators, seed=config.seed
        ),
        "lof": LOFBaseline(n_neighbors=config.lof_n_neighbors),
        "knn": KNNBaseline(n_neighbors=config.knn_n_neighbors),
    }
